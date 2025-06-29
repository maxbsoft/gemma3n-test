#!/usr/bin/env python3
"""
Setup script for Gemma 3n API Server
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command"""
    print(f"Running: {description or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    
    return True

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python requirements...")
    
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install requirements")
        return False
    
    print("✅ Requirements installed successfully")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA Available")
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - will run on CPU (much slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def setup_environment():
    """Setup environment file"""
    print("\n🔧 Setting up environment...")
    
    env_path = Path("../.env")
    
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    print("Creating .env file template...")
    
    env_content = """# Hugging Face Token (required)
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_huggingface_token_here

# Optional: Model cache directory
# CACHE_DIR=./model_cache

# Optional: Server configuration
# HOST=0.0.0.0
# PORT=8000
"""
    
    with open(env_path, "w") as f:
        f.write(env_content)
    
    print(f"✅ Created {env_path}")
    print("⚠️  Please edit .env and add your Hugging Face token")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing imports...")
    
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("requests", "Requests"),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not found")
            all_good = False
    
    return all_good

def main():
    """Main setup function"""
    print("🚀 Setting up Gemma 3n API Server")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import. Please check your installation.")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Setup environment
    setup_environment()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit ../.env and add your Hugging Face token")
    print("2. Accept Gemma license at: https://huggingface.co/google/gemma-3n-e4b-it")
    print("3. Start the server: python start_server.py")
    print("4. Check examples: python examples/basic_usage.py")
    
    print("\nAPI will be available at:")
    print("  - http://localhost:8000 (API)")
    print("  - http://localhost:8000/docs (Documentation)")

if __name__ == "__main__":
    main() 