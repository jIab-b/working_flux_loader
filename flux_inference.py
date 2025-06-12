import torch
from diffusers import FluxPipeline
from accelerate import disk_offload
import gc
import os

def load_flux_with_disk_offload():
    model_id = "black-forest-labs/FLUX.1-dev"
    offload_dir = "./flux_offload"
    
    # Create offload directory
    os.makedirs(offload_dir, exist_ok=True)
    

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None,  
        low_cpu_mem_usage=True,
    )
    


    pipe.text_encoder.to("cuda:0")
    pipe.text_encoder_2.to("cuda:0")

    pipe.vae.to("cuda:0")
    

    print("Offloading transformer to disk...")
    disk_offload(
        model=pipe.transformer,
        offload_dir=os.path.join(offload_dir, "transformer"),
        execution_device="cuda:0",  # Will be moved to GPU when needed

    )
    
    return pipe

def main():

    pipe = load_flux_with_disk_offload()
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    

    torch.cuda.empty_cache()
    gc.collect()
    
    prompt = "a fantasy landscape with mountains and rivers, trending on artstation"
    
    # Generate with conservative settings
    image = pipe(
        prompt,
        num_inference_steps=5,
        guidance_scale=3.0,
        height=512,
        width=512,
        max_sequence_length=128,
    ).images[0]

    image.save("out.png")
       


if __name__ == "__main__":
    main()
