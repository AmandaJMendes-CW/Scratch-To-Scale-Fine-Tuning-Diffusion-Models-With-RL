#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel
import json

# Add the scripts directory to the path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from custom_ddim_scheduler import CustomDDIMScheduler


def load_model_and_scheduler(model_dir: str, device: str = "cuda"):
    """
    Load the fine-tuned model and scheduler from the specified directory.
    
    Args:
        model_dir: Path to the directory containing the saved model
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded UNet2DModel
        scheduler: CustomDDIMScheduler
        config: Training configuration
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load the model
    model = UNet2DModel.from_pretrained(model_dir).to(device)
    model.eval()
    
    # Load the scheduler
    scheduler_path = os.path.join(model_dir, "scheduler")
    if os.path.exists(scheduler_path):
        # Load the saved scheduler
        scheduler = CustomDDIMScheduler.from_pretrained(scheduler_path, use_safetensors=True)
        print("Loaded saved scheduler")
    else:
        # Fallback to base model scheduler if not saved
        scheduler = CustomDDIMScheduler.from_pretrained("google/ddpm-celebahq-256", use_safetensors=True)
        print("Using base model scheduler (saved scheduler not found)")
    
    # Load training configuration if available
    config_path = os.path.join(model_dir, "training_args.json")
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return model, scheduler, config


def generate_images(
    model: UNet2DModel,
    scheduler: CustomDDIMScheduler,
    num_images: int = 4,
    num_inference_steps: int = 50,
    device: str = "cuda",
    seed: int = None
) -> list[Image.Image]:
    """
    Generate images using the loaded model.
    
    Args:
        model: The diffusion model
        scheduler: The DDIM scheduler
        num_images: Number of images to generate
        num_inference_steps: Number of denoising steps
        device: Device to run inference on
        seed: Random seed for reproducible generation
    
    Returns:
        List of PIL Images
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Set up scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    # Generate noise
    n_channels = model.config.in_channels
    image_size = model.config.sample_size
    latents = torch.randn(
        (num_images, n_channels, image_size, image_size),
        device=device
    )
    
    # Denoising loop
    with torch.no_grad():
        for t in scheduler.timesteps:
            # Predict noise
            pred_noise = model(latents, t).sample
            
            # Step the scheduler
            scheduler_output, _ = scheduler.step(pred_noise, t, latents, eta=1.0)
            latents = scheduler_output.prev_sample
    
    # Convert to images
    images = []
    latents = latents.cpu().permute(0, 2, 3, 1)
    latents = ((latents + 1.0) * 127.5).numpy().astype(np.uint8)
    
    for i in range(num_images):
        image = Image.fromarray(latents[i])
        images.append(image)
    
    return images


def main():
    parser = argparse.ArgumentParser(description='Generate images using a fine-tuned diffusion model')
    parser.add_argument('model_dir', type=str, help='Path to the saved model directory')
    parser.add_argument('--num_images', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible generation')
    parser.add_argument('--output_dir', type=str, default='generated_images', help='Output directory for images')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run on')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Loading model from {args.model_dir}...")
    try:
        model, scheduler, config = load_model_and_scheduler(args.model_dir, args.device)
        print("Model loaded successfully!")
        
        if config:
            print(f"Model was trained with {config.get('inference_timesteps', 'unknown')} inference steps")
            if args.num_inference_steps != config.get('inference_timesteps', 50):
                print(f"Note: Using {args.num_inference_steps} steps instead of training default {config.get('inference_timesteps', 50)}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Generating {args.num_images} images...")
    try:
        images = generate_images(
            model=model,
            scheduler=scheduler,
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            seed=args.seed
        )
        
        # Save images
        os.makedirs(args.output_dir, exist_ok=True)
        for i, image in enumerate(images):
            filename = f"generated_image_{i:03d}.png"
            filepath = os.path.join(args.output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")
        
        print(f"Successfully generated {len(images)} images in {args.output_dir}")
        
    except Exception as e:
        print(f"Error generating images: {e}")


if __name__ == "__main__":
    main()