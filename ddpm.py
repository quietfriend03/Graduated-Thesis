import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_traning_steps: 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        self.betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_traning_steps, dtype=np.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas, axis=0) # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.ones = torch.Tensor(1.0)
        
        self.generator = generator
        self.num_traning_steps = num_traning_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_traning_steps)[::-1].copy())
        
    def set_inferences(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_traning_steps / self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        
    def get_prev_timesteps(self, timesteps: int) -> int:
        prev_timesteps = timesteps - (self.num_traning_steps // self.num_inference_steps) 
        return prev_timesteps
    
    def get_variance(self, timesteps: int) -> torch.Tensor:
        prev_t = self.get_prev_timesteps(timesteps)
        alpha_prod_t = self.alpha_cumprod[timesteps]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.ones
        
        current_beta_t = 1 - alpha_prod_t/alpha_prod_prev_t
        
        # Formula (7) in the paper
        variance = ((1 - alpha_prod_prev_t) / (1 - alpha_prod_t)) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        
        return variance
    
    def set_strength(self, strength = 1.0):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step        
        
    def step(self, timesteps: int, latents: torch.FloatTensor, model_output: torch.FloatTensor):
        t = timesteps
        prev_t = self.get_prev_timesteps(t)
        alpha_prod_t = self.alpha_cumprod[timesteps]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.ones
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t
        
        pred_original_sample = ((latents - beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5) 
        
        # Compute the coefficient for the prediction and current sample
        pred_original_sample_coeff = (alpha_prod_t ** 0.5 * current_beta_t) / beta_prod_t
        
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_prev_t / beta_prod_t
        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        
        if t > 0:
            device = model_output.device
            noise = torch.randn(latents.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self.get_variance(t)) * noise
            
        # N(0, 1) => N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0, 1)
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample
        
        
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)
        
        sqrt_alphas_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.flatten()
        
        while len(sqrt_alphas_cumprod.shape) < len(original_samples.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            
        sqrt_one_minus_alphas_cumprod = (1 - alpha_cumprod[timesteps]) ** 0.5 # Std
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.flatten()
        
        while(len(sqrt_one_minus_alphas_cumprod.shape) < len(original_samples.shape)):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
            
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noise_sample = (sqrt_alphas_cumprod * original_samples) + (sqrt_one_minus_alphas_cumprod) * noise
        return noise_sample
    
    