from transformers import pipeline, logging
import ImageReward as RM
import numpy as np
import torch
import PIL

def image_reward(pil_images, prompt = "sharp, photo-realistic portrait of a human face, no noise artifacts"):
    if not hasattr(image_reward, "_init"):
        image_reward.image_reward_model = RM.load("ImageReward-v1.0", device="cpu")
        image_reward._init = True
        
    return image_reward.image_reward_model.score(prompt, pil_images)

def gender_reward(pil_images):
    if not hasattr(gender_reward, "_init"):
        logging.set_verbosity_error()
        gender_reward.pipe = pipeline("image-classification", model="rizvandwiki/gender-classification", device="cpu")
        gender_reward._init = True
        
    classification = gender_reward.pipe(pil_images)

    scores = []
    for preds in classification:
        male_prob = next(p["score"] for p in preds if p["label"] == "male")
        if male_prob >= 0.8 or male_prob <= 0.2:
            scores.append(male_prob)
        else:
            scores.append(0.0)
    return scores

def reward_function(latents_batch):
    # Move latents to CPU to save GPU memory
    latents_batch = latents_batch.to("cpu")

    # Convert latents batch to list of PIL images
    image_processed = latents_batch.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)
    images = [PIL.Image.fromarray(image_processed[i]) for i in range(image_processed.shape[0])]
    
    ir_person = image_reward(images, "a natural, high-quality portrait photograph of a person with realistic facial features, normal hair color, natural expression, and clean background")
    #ir_man = image_reward(images, "sharp, photo-realistic portrait of a man, no noise artifacts")
    sex_score = gender_reward(images)

    sex_score = torch.tensor(sex_score)
    ir_person = torch.tensor(ir_person)
    
    sex_score_binary = (sex_score > 0.5).float()
    
    total_score = ir_person + 2*sex_score_binary

    # Return all metrics for logging
    return total_score, {
        'ir_person': ir_person,
        'sex_score': sex_score,
        'sex_score_binary': sex_score_binary
    }