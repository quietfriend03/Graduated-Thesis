from clip.clip import CLIP
from vae.vae_encoder import Encoder
from vae.vae_decoder import Decoder
from diffusion.diffusion import Diffusion

import converter

def load_model_from_standard_weights(ckpt_path, device):
    state_dict = converter.load_from_standard_weights(ckpt_path, device)
    
    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    
    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)
    
    return {
        "clip": clip, 
        "encoder": encoder, 
        "decoder": decoder, 
        "diffusion": diffusion
    }