"""Approach C: LoRA adaptation and InstructPix2Pix training objective.

The notebook owns shared data, validation and resumable training orchestration.
Target latents are VAE-scaled; source-image conditioning latents MUST remain unscaled.
"""
import torch


def configure_adapter(pipe, rank):
    from peft import LoraConfig, get_peft_model
    assert pipe.unet.config.in_channels == 8
    pipe.vae.requires_grad_(False).eval().to(dtype=torch.float32)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.unet.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian',
                       target_modules=['to_q', 'to_k', 'to_v', 'to_out.0', 'conv_in'])
    pipe.unet = get_peft_model(pipe.unet, config)
    pipe.unet.enable_gradient_checkpointing()
    for param in pipe.unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return pipe


def noise_prediction_loss(pipe, scheduler, embeddings, batch, *, device, dtype,
                          conditioning_dropout=.05, snr_gamma=5., training=True,
                          fixed_seed=None, timestep=None):
    generator = None if fixed_seed is None else torch.Generator(device=device).manual_seed(fixed_seed)
    source = batch['source'].to(device, dtype=torch.float32)
    target = batch['target'].to(device, dtype=torch.float32)
    with torch.no_grad():
        posterior = pipe.vae.encode(target).latent_dist
        clean = posterior.sample(generator=generator) if training else posterior.mode()
        clean = (clean * pipe.vae.config.scaling_factor).to(dtype)
        source_latents = pipe.vae.encode(source).latent_dist.mode().to(dtype)
    size = clean.shape[0]
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=dtype)
    timesteps = (torch.randint(scheduler.config.num_train_timesteps, (size,), device=device, generator=generator)
                 if timestep is None else torch.full((size,), timestep, device=device, dtype=torch.long))
    noisy = scheduler.add_noise(clean, noise, timesteps)
    text = embeddings[:1].expand(size, -1, -1)
    if training:
        p = conditioning_dropout
        draw = torch.rand(size, device=device)
        text = torch.where((draw < 2*p).view(size, 1, 1), embeddings[1:].expand_as(text), text)
        source_latents = source_latents * (~((draw >= p) & (draw < 3*p))).view(size, 1, 1, 1)
    with torch.autocast(device_type=str(device).split(':')[0], dtype=dtype, enabled=dtype != torch.float32):
        prediction = pipe.unet(torch.cat([noisy, source_latents], dim=1), timesteps,
                               encoder_hidden_states=text).sample
    prediction_type = scheduler.config.prediction_type
    if prediction_type == 'epsilon':
        objective = noise
    elif prediction_type == 'v_prediction':
        objective = scheduler.get_velocity(clean, noise, timesteps)
    else:
        raise ValueError(prediction_type)
    loss = (prediction.float()-objective.float()).square().mean(dim=(1, 2, 3))
    alpha = scheduler.alphas_cumprod.to(device)[timesteps]
    snr = alpha/(1-alpha).clamp_min(1e-8)
    weights = snr.clamp(max=snr_gamma)/(snr+(prediction_type == 'v_prediction')).clamp_min(1e-8)
    return (loss*weights).mean()
