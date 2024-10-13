import os
import argparse
import zipfile
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch.optim import AdamW
import json

def huggingface_login(token):
    os.system(f"huggingface-cli login --token {token}")

def download_image(url, output_dir):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image_name = os.path.basename(url.split('?')[0])
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {image_name}")
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def download_images_from_list(file_path, output_dir):
    if not os.path.exists(file_path):
        print(f"Image URL list file {file_path} does not exist.")
        exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_path, 'r') as f:
        urls = f.readlines()
    
    for url in urls:
        url = url.strip()
        if url:
            download_image(url, output_dir)

def generate_captions(input_dir, output_dir, model_name='Salesforce/blip-image-captioning-base'):
    print(f"Generating captions for images in: {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        exit(1)

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        exit(1)

    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file = os.path.splitext(image_name)[0] + ".txt"
            with open(os.path.join(output_dir, caption_file), "w") as f:
                f.write(caption)
            print(f"Caption generated for {image_name}: {caption}")
        except Exception as e:
            print(f"Failed to generate caption for {image_name}: {e}")
    
    print(f"Captions generated and saved in {output_dir}")

def train_model(data_dir, checkpoint_path, trigger_word, model_name, hf_token, config):
    print("Loading Stable Diffusion Pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    print("Setting up optimizer...")
    optimizer = AdamW(params=pipe.unet.parameters(), lr=config['unet_lr'])

    print("Starting training...")
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        # Placeholder for training logic
        # You need to implement the actual training steps here
        for step in range(config['max_train_steps']):
            # Simulate a training step
            if step % 100 == 0:
                print(f"Training step {step}/{config['max_train_steps']}")
            # Add actual training code here

    print("Training completed.")
    print(f"Saving trained model to {model_name}...")
    pipe.save_pretrained(model_name)

    print("Logging into Hugging Face...")
    huggingface_login(hf_token)

    print("Uploading model to Hugging Face...")
    os.system(f"git init")
    os.system(f"git remote add origin https://huggingface.co/{model_name}")
    os.system(f"git add . && git commit -m 'Add trained model' && git push origin main")
    
    print("Model uploaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--image_urls", type=str, default="image_urls.txt", help="Path to the text file containing image URLs")
    parser.add_argument("--checkpoint_url", type=str, required=True, help="URL of the base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save trained model")
    parser.add_argument("--trigger_word", type=str, default="Navodix", help="Trigger word for the model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--config", type=str, required=True, help="JSON file with hyperparameters")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model repository (e.g., your-username/YourModelName)")

    args = parser.parse_args()

    # Step 1: Download images from URLs
    download_images_from_list(args.image_urls, "./dataset/images")

    # Step 2: Download the base model checkpoint
    download_checkpoint(args.checkpoint_url, "./base_model.safetensors")

    # Step 3: Generate captions for images
    generate_captions("./dataset/images", "./dataset/captions")

    # Step 4: Load configuration
    with open(args.config) as f:
        config = json.load(f)

    # Step 5: Train the model
    train_model("./dataset", "./base_model.safetensors", args.trigger_word, args.model_name, args.hf_token, config)
