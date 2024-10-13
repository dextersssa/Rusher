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

def download_dataset(url, output_dir):
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download dataset. Status code: {response.status_code}")
        exit(1)
    
    dataset_zip_path = os.path.join(output_dir, "dataset.zip")
    with open(dataset_zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print(f"Extracting dataset to {output_dir}...")
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Dataset downloaded and extracted to {output_dir}")
    print("Contents of the extracted directory:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

def download_checkpoint(checkpoint_url, output_path):
    print(f"Downloading checkpoint from {checkpoint_url}...")
    response = requests.get(checkpoint_url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download checkpoint. Status code: {response.status_code}")
        exit(1)
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Checkpoint downloaded to {output_path}")

def find_image_directory(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            return root
    return None

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
    parser.add_argument("--data_url", type=str, required=True, help="URL of the dataset zip file")
    parser.add_argument("--checkpoint_url", type=str, required=True, help="URL of the base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save trained model")
    parser.add_argument("--trigger_word", type=str, default="Navodix", help="Trigger word for the model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--config", type=str, required=True, help="JSON file with hyperparameters")

    args = parser.parse_args()

    # Step 1: Download and extract the dataset
    download_dataset(args.data_url, "./dataset")

    # Step 2: Download the base model checkpoint
    download_checkpoint(args.checkpoint_url, "./base_model.safetensors")

    # Step 3: Find the image directory
    image_dir = find_image_directory("./dataset")
    if image_dir is None:
        print("No directory with images found in the dataset.")
        exit(1)
    
    print(f"Using image directory: {image_dir}")

    # Step 4: Generate captions for images
    generate_captions(image_dir, "./dataset/captions")

    # Step 5: Load configuration
    with open(args.config) as f:
        config = json.load(f)

    # Step 6: Train the model
    train_model("./dataset", "./base_model.safetensors", args.trigger_word, args.output_dir, args.hf_token, config)
