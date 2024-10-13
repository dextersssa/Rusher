import os
import argparse
import zipfile
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch.optim import AdamW

def huggingface_login(token):
    os.system(f"huggingface-cli login --token {token}")

def download_dataset(url, output_dir):
    response = requests.get(url)
    with open("dataset.zip", 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
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
    response = requests.get(checkpoint_url)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Checkpoint downloaded to {output_path}")

def generate_captions(input_dir, output_dir, model_name='Salesforce/blip-image-captioning-base'):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        caption_file = os.path.splitext(image_name)[0] + ".txt"
        with open(os.path.join(output_dir, caption_file), "w") as f:
            f.write(caption)
    
    print(f"Captions generated and saved in {output_dir}")

def train_model(data_dir, checkpoint_path, trigger_word, model_name, hf_token, config):
    pipe = StableDiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    optimizer = AdamW(params=pipe.unet.parameters(), lr=config['unet_lr'])

    for epoch in range(config['epochs']):
        for step, data in enumerate(os.listdir(data_dir)):
            if step >= config['max_train_steps']:
                break
            # Add your training logic here

        print(f"Epoch {epoch+1}/{config['epochs']} completed.")

    pipe.save_pretrained(model_name)

    huggingface_login(hf_token)
    os.system(f"git init")
    os.system(f"git remote add origin https://huggingface.co/{model_name}")
    os.system(f"git add . && git commit -m 'Add trained model' && git push origin main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data_url", type=str, required=True, help="URL of the dataset zip file")
    parser.add_argument("--checkpoint_url", type=str, required=True, help="URL of the base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save trained model")
    parser.add_argument("--trigger_word", type=str, default="Navodix", help="Trigger word for the model")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument("--config", type=str, required=True, help="JSON file with hyperparameters")

    args = parser.parse_args()

    download_dataset(args.data_url, "./dataset")
    download_checkpoint(args.checkpoint_url, "./base_model.safetensors")

    # Find the correct image directory
    image_dir = "./dataset"
    for root, dirs, files in os.walk("./dataset"):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            image_dir = root
            break

    print(f"Using image directory: {image_dir}")
    generate_captions(image_dir, "./dataset/captions")

    import json
    with open(args.config) as f:
        config = json.load(f)

    train_model("./dataset", "./base_model.safetensors", args.trigger_word, args.output_dir, args.hf_token, config)
