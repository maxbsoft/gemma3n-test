"""
Basic usage examples for Gemma 3n API
"""
import asyncio
import requests
import base64
from pathlib import Path

# API configuration
API_BASE = "http://localhost:8000/v1"

def test_basic_chat():
    """Test basic text chat completion"""
    print("=== Basic Text Chat ===")
    
    payload = {
        "model": "gemma-3n-e2b-quantized",
        "messages": [
            {"role": "user", "content": "Tell me a short story about a robot learning to paint"}
        ],
        "max_tokens": 150,
        "temperature": 0.8
    }
    
    response = requests.post(f"{API_BASE}/chat/completions", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']['completion_tokens']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_multimodal_with_url():
    """Test multimodal chat with image URL"""
    print("\n=== Multimodal Chat with Image URL ===")
    
    payload = {
        "model": "gemma-3n-e2b-quantized",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image? Describe it in detail."},
                    {"type": "image", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}
                ]
            }
        ],
        "max_tokens": 200
    }
    
    response = requests.post(f"{API_BASE}/chat/completions", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Image description: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_multimodal_with_base64():
    """Test multimodal chat with base64 image"""
    print("\n=== Multimodal Chat with Base64 Image ===")
    
    # Create a simple test image if it doesn't exist
    test_image_path = Path("../test_image.png")
    if not test_image_path.exists():
        print("Test image not found, creating one...")
        from PIL import Image
        img = Image.new('RGB', (200, 200), (100, 150, 200))
        img.save(test_image_path)
    
    # Read and encode image
    with open(test_image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    payload = {
        "model": "gemma-3n-e2b-quantized",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colors do you see in this image?"},
                    {"type": "image", "base64": image_base64}
                ]
            }
        ],
        "max_tokens": 100
    }
    
    response = requests.post(f"{API_BASE}/chat/completions", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Color description: {result['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_streaming():
    """Test streaming response"""
    print("\n=== Streaming Response ===")
    
    payload = {
        "model": "gemma-3n-e2b-quantized",
        "messages": [
            {"role": "user", "content": "Write a creative poem about artificial intelligence and the future"}
        ],
        "max_tokens": 200,
        "stream": True
    }
    
    with requests.post(f"{API_BASE}/chat/completions", json=payload, stream=True) as response:
        if response.status_code == 200:
            print("Streaming response:")
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        data = line_text[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            if chunk["choices"][0]["delta"].get("content"):
                                print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
                        except:
                            continue
            print("\n")
        else:
            print(f"Error: {response.status_code} - {response.text}")

def test_model_switching():
    """Test switching between models"""
    print("\n=== Model Switching ===")
    
    # List available models
    response = requests.get(f"{API_BASE}/models")
    if response.status_code == 200:
        models = response.json()
        print("Available models:")
        for model in models["data"]:
            print(f"  - {model['id']}")
    
    # Load a different model
    payload = {"model": "gemma-3n-e4b-full"}
    response = requests.post(f"{API_BASE}/models/load", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model loading: {result['message']}")
        if result['success']:
            print(f"New model info: {result['model_info']}")
    else:
        print(f"Error loading model: {response.status_code} - {response.text}")

def test_generation_presets():
    """Test different generation presets"""
    print("\n=== Generation Presets ===")
    
    prompt = "Explain quantum computing"
    presets = ["creative", "balanced", "precise", "deterministic"]
    
    for preset in presets:
        print(f"\n--- {preset.upper()} PRESET ---")
        payload = {
            "model": "gemma-3n-e2b-quantized",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "preset": preset
        }
        
        response = requests.post(f"{API_BASE}/chat/completions", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(result['choices'][0]['message']['content'])
        else:
            print(f"Error: {response.status_code} - {response.text}")

def check_server_status():
    """Check server health and status"""
    print("\n=== Server Status ===")
    
    # Health check
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"Server status: {health['status']}")
        print(f"GPU available: {health['gpu_available']}")
        if health['gpu_memory']:
            print(f"GPU memory: {health['gpu_memory']}")
        if health['current_model']:
            print(f"Current model: {health['current_model']}")
    
    # Detailed status
    response = requests.get(f"{API_BASE}/status")
    if response.status_code == 200:
        status = response.json()
        print(f"\nDetailed status:")
        print(f"  Server: {status['server_status']}")
        print(f"  Available models: {status['available_models']}")
        if status['gpu_info']:
            print(f"  GPU: {status['gpu_info']['name']}")
            print(f"  GPU Memory: {status['gpu_info']['memory_allocated']}")

if __name__ == "__main__":
    print("🚀 Testing Gemma 3n API")
    print("Make sure the server is running on http://localhost:8000")
    
    try:
        # Check if server is running
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Start it with: python main.py")
        exit(1)
    
    print("✅ Server is running, starting tests...\n")
    
    # Run tests
    check_server_status()
    test_basic_chat()
    test_multimodal_with_url()
    # test_multimodal_with_base64()  # Uncomment if you have test image
    test_streaming()
    test_generation_presets()
    # test_model_switching()  # Uncomment to test model switching
    
    print("\n🎉 All tests completed!") 