import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from DDPM import DDPMSampler
import numpy as np

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(
        prompt: str, 
        unconditional_prompt: str,   # Negative prompt or empty string
        input_image=None, 
        strength=0.7,                # How much attention to pay for the initial image (more strength, more noise)
        do_cfg=True,                 # Whether to use classifier-free guidance
        cfg_scale=8,                 # The weight how much model pay attention to prompt [1, 40]
        sampler_name="ddpm",         # Which sampler to use
        n_inference_steps=100,       # Number of inference steps
        models={}, 
        seed=42,
        device=None,                 # Which device to create tensor 
        idle_device=None,            # Load model to CUDA and if don't need it will move to CPU
        tokenizer=None,
    ):
    with torch.no_grad():
        if not(0 < strength <= 1):
            raise ValueError("strength must be in [0, 1]")
        
        if idle_device:
            to_idle_device = lambda x: x.to(idle_device)
        else:
            to_idle_device = lambda x: x
            
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
            
        clip = models['clip']
        clip.to(device)
        
        if do_cfg:
            # Converting prompt to token (Conditional & Unconditional)
            
            conditional_token = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77)["input_ids"]
            # (Batch, Sequence Length)
            conditional_token = torch.tensor(conditional_token, dtype=torch.long, device=device)
            # (Batch, Sequence Length) => (Batch, Sequence Length, Dimension)
            conditional_context = clip(conditional_token)
            
            unconditional_token = tokenizer.batch_encode_plus([unconditional_prompt], padding="max_length", max_length=77)["input_ids"]
            # (Batch, Sequence Length)
            unconditional_token = torch.tensor(unconditional_token, dtype=torch.long, device=device)
            # (Batch, Sequence Length) => (Batch, Sequence Length, Dimension)
            unconditional_context = clip(unconditional_token)
            
            # (2, Sequence Length, Dimension) = (2, 77, 768)
            context = torch.cat([conditional_context, unconditional_context]) 
        
        else: 
            # Converting into a list of tokens
            token = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77)["input_ids"]
            token = torch.tensor(token, dtype=torch.long, device=device)
            # (1, Sequence Length, Dimension) = (1, 77, 768)
            context = clip(token)
        to_idle_device(clip)
            
            
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not supported")
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        
        if input_image:
            encoder = models['encoder']
            encoder.to(device)
            
            # Change image to tensor
            input_image_tensor = input_image.resize((WIDTH, HEIGHT)) 
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) => (Batch_size, Height, Width, Channel) => (Batch_size ,Channel, Height, Width)
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)
            
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # Run the image through the VAE encoder
            latents = encoder(input_image_tensor, encoder_noise)
            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle_device(encoder)
        else:
            # Text-to-Image, random noise
            latents = torch.randn(latent_shape, generator=generator, device=device)
            
        diffusion = models['diffusion']
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        for i, timesteps in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timesteps).to(device)
            
            # (Batch_size, 4, Latent Hight,Latent Width)
            model_input = latents
            
            if do_cfg:
                # (Batch_size, 4,Latent Height,Latent Width) => (2 * Batch_size, 4,Latent Height,Latent Width)
                model_input = model_input.repeat(2, 1, 1, 1)
            
            # Predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)
            
            if do_cfg:
                conditional_output, unconditional_output = model_output.chunk(2)
                model_output = cfg_scale * (conditional_output - unconditional_output) + unconditional_output
                
            latents = sampler.step(timesteps, latents, model_output)
                
        to_idle_device(diffusion)
        
        decoder = models['decoder']
        decoder.to(device)
        
        images = decoder(latents)
        to_idle_device(decoder)
        
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_size, Channel, Height, Width) => (Batch_size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
    def rescale(x, old_range, new_range, clamp=False):
        old_min, old_max = old_range
        new_min, new_max = new_range
        
        x -= old_min
        x *= (new_max - new_min) / (old_max - old_min)
        x += new_min
        
        if clamp:
            x = x.clamp(new_min, new_max)
            
        return x
    
    def get_time_embedding(timesteps):
        # (160, )
        frequency = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=device) / 160)
        
        # (1, 160)
        x = torch.tensor([timesteps], dtype=torch.float32)[:, None] * frequency[None]
        # (1, 160 * 2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        
    