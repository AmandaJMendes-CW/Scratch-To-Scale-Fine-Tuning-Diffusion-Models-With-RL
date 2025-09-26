#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import UNet2DModel
from accelerate import Accelerator
from accelerate.logging import get_logger
import json
import math
import logging

# Add the scripts directory to the path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from custom_ddim_scheduler import CustomDDIMScheduler


def load_model_and_scheduler(model_dir: str, accelerator: Accelerator, logger=None):
    """
    Load the fine-tuned model and scheduler from the specified directory.
    
    Args:
        model_dir: Path to the directory containing the saved model
        accelerator: Accelerator instance for multi-GPU support
        logger: Optional logger for logging messages
    
    Returns:
        model: Loaded UNet2DModel
        scheduler: CustomDDIMScheduler
        config: Training configuration
    """
    if logger is None:
        logger = get_logger(__name__)

    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logger.info(f"Loading UNet2DModel from {model_dir} ...")
    model = UNet2DModel.from_pretrained(model_dir).to(accelerator.device)
    model.eval()
    logger.info("Model loaded and set to eval mode.")

    # Prepare model for multi-GPU inference
    model = accelerator.prepare(model)
    logger.info("Model prepared for accelerator.")

    # Load the scheduler
    scheduler_path = os.path.join(model_dir, "scheduler")
    if os.path.exists(scheduler_path):
        scheduler = CustomDDIMScheduler.from_pretrained(scheduler_path, use_safetensors=True)
        logger.info("Loaded saved scheduler from model directory.")
    else:
        scheduler = CustomDDIMScheduler.from_pretrained("google/ddpm-celebahq-256", use_safetensors=True)
        logger.warning("Saved scheduler not found, using base model scheduler.")

    # Load training configuration if available
    config_path = os.path.join(model_dir, "training_args.json")
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded training config from {config_path}")
    else:
        logger.warning("No training_args.json found in model directory.")

    return model, scheduler, config


def generate_images_batch(
    model: UNet2DModel,
    scheduler: CustomDDIMScheduler,
    batch_size: int,
    num_inference_steps: int,
    accelerator: Accelerator,
    generator: torch.Generator = None,
    logger=None
) -> list[Image.Image]:
    """
    Generate a batch of images using the loaded model with multi-GPU support.
    
    Args:
        model: The diffusion model
        scheduler: The DDIM scheduler
        batch_size: Number of images to generate in this batch
        num_inference_steps: Number of denoising steps
        accelerator: Accelerator instance for multi-GPU support
        generator: Optional generator for reproducible generation
        logger: Optional logger for logging messages
    
    Returns:
        List of PIL Images
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Setting scheduler timesteps to {num_inference_steps} on device {accelerator.device}")
    scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device)
    
    n_channels = model.module.config.in_channels if hasattr(model, 'module') else model.config.in_channels
    image_size = model.module.config.sample_size if hasattr(model, 'module') else model.config.sample_size

    logger.info(f"Generating noise latents: batch_size={batch_size}, channels={n_channels}, size={image_size}")
    latents = torch.randn(
        (batch_size, n_channels, image_size, image_size),
        device=accelerator.device,
        generator=generator
    )
    
    logger.info(f"Starting denoising loop for {num_inference_steps} steps.")
    with torch.no_grad():
        for t in scheduler.timesteps:
            pred_noise = model(latents, t).sample
            scheduler_output, _ = scheduler.step(pred_noise, t, latents, eta=1.0, generator=generator)
            latents = scheduler_output.prev_sample

    logger.info("Denoising complete. Gathering latents from all processes.")
    gathered_latents = accelerator.gather(latents)
    
    images = []
    if accelerator.is_main_process:
        logger.info("Converting gathered latents to images on main process.")
        gathered_latents = gathered_latents.cpu().permute(0, 2, 3, 1)
        gathered_latents = ((gathered_latents + 1.0) * 127.5).numpy().astype(np.uint8)
        
        for i in range(gathered_latents.shape[0]):
            image = Image.fromarray(gathered_latents[i])
            images.append(image)
        logger.info(f"Converted {len(images)} images from latents.")

    return images


def main():
    parser = argparse.ArgumentParser(description='Generate images using a fine-tuned diffusion model with multi-GPU support')
    parser.add_argument('--model_dir', type=str, help='Path to the saved model directory')
    parser.add_argument('--num_images', type=int, default=16, help='Total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible generation')
    parser.add_argument('--output_dir', type=str, default='generated_images', help='Output directory for images')
    
    args = parser.parse_args()
    
    accelerator = Accelerator()
    logger = get_logger(__name__)
    logger.setLevel(logging.DEBUG)

    if accelerator.is_main_process:
        logger.info(f"Using {accelerator.num_processes} GPU(s) for inference")
        logger.info(f"Loading model from {args.model_dir}...")

    try:
        model, scheduler, config = load_model_and_scheduler(args.model_dir, accelerator, logger=logger)
        if accelerator.is_main_process:
            logger.info("Model loaded successfully!")
            if config:
                logger.info(f"Model was trained with {config.get('inference_timesteps', 'unknown')} inference steps")
                if args.num_inference_steps != config.get('inference_timesteps', 50):
                    logger.info(f"Note: Using {args.num_inference_steps} steps instead of training default {config.get('inference_timesteps', 50)}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    num_batches = math.ceil(args.num_images / (args.batch_size * accelerator.num_processes))
    if accelerator.is_main_process:
        logger.info(f"Generating {args.num_images} images using {num_batches} batches per GPU (batch size {args.batch_size} per GPU)...")
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        logger.debug("Not main process, will not save images.")

    all_images = []

    try:
        for i in range(num_batches):
            if accelerator.is_main_process:
                logger.info(f"Generating batch {i+1}/{num_batches} ...")
            generator = None
            if args.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + i + accelerator.process_index)
                if accelerator.is_main_process:
                    logger.info(f"Using seed {args.seed + i + accelerator.process_index} for batch {i+1}")

            batch_images = generate_images_batch(
                model=model,
                scheduler=scheduler,
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                accelerator=accelerator,
                generator=generator,
                logger=logger
            )

            if accelerator.is_main_process and batch_images:
                logger.info(f"Extending all_images with {len(batch_images)} images from batch {i+1}")
                all_images.extend(batch_images)

        if accelerator.is_main_process:
            all_images = all_images[:args.num_images]
            logger.info(f"Saving {len(all_images)} images to {args.output_dir}")
            for i, image in enumerate(all_images):
                filename = f"generated_image_{i:03d}.png"
                filepath = os.path.join(args.output_dir, filename)
                image.save(filepath)
                logger.debug(f"Saved image {filepath}")
            logger.info(f"Successfully generated {len(all_images)} images in {args.output_dir}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info("All processes synchronized. Generation complete.")

    except Exception as e:
        logger.error(f"Error generating images: {e}")


if __name__ == "__main__":
    main()