import os
import torch
from transformers.pipelines import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optimization settings
torch.set_float32_matmul_precision('high')

# Get HF token from environment
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

print("Setting up Gemma 3n pipeline...")
print(f"Using token: {hf_token[:10]}...")

# Create pipeline with authentication
pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3n-E4B-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
    token=hf_token  # Add token for model download
)

print("Model loaded successfully!")

# Test with image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/ariG23498/demo-data/resolve/main/airplane.jpg"},
            {"type": "text", "text": "Describe this image"}
        ]
    }
]

print("Generating response...")
output = pipe(messages, max_new_tokens=32)
print("\nResult:")
print(output[0]["generated_text"][-1]["content"])

# Test with text only
print("\n" + "="*50)
print("Testing text-only generation:")

text_messages = [
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "Write a short poem about AI"}
        ]
    }
]

text_output = pipe(text_messages, max_new_tokens=50)
print("Text generation result:")
print(text_output[0]["generated_text"][-1]["content"]) 