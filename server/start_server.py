#!/usr/bin/env python3
"""
Start script for Gemma 3n API Server
"""
import argparse
import os
import sys
import uvicorn
import torch
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

from config import config, ModelConfig

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    
    # Check HF_TOKEN
    if not config.HF_TOKEN:
        issues.append("❌ HF_TOKEN not found in environment variables")
        issues.append("   Please set your Hugging Face token in ../.env file")
    
    # Check CUDA
    if not torch.cuda.is_available():
        issues.append("⚠️  CUDA not available - will run on CPU (much slower)")
    else:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 6:
            issues.append("⚠️  GPU has less than 6GB memory - use quantized models only")
    
    # Check Python packages
    try:
        import transformers
        import fastapi
        import uvicorn
        print(f"✅ FastAPI version: {fastapi.__version__}")
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError as e:
        issues.append(f"❌ Missing package: {e.name}")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Start Gemma 3n API Server")
    
    parser.add_argument(
        "--host", 
        default=config.HOST,
        help=f"Host to bind to (default: {config.HOST})"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=config.PORT,
        help=f"Port to bind to (default: {config.PORT})"
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        choices=list(ModelConfig.get_all_models().keys()),
        help=f"Model to load on startup (default: {config.DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.WORKERS,
        help=f"Number of worker processes (default: {config.WORKERS})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip requirement checks"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Gemma 3n API Server")
    print("=" * 50)
    
    # Check requirements
    if not args.no_check:
        print("Checking requirements...")
        issues = check_requirements()
        
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(issue)
            
            if any("❌" in issue for issue in issues):
                print("\n❌ Critical issues found. Please fix them before starting.")
                sys.exit(1)
            else:
                print("\n⚠️  Warnings found but server can start.")
    
    # Update config with command line args
    config.DEFAULT_MODEL = args.model
    
    print(f"\nServer configuration:")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Default model: {args.model}")
    print(f"  Workers: {args.workers}")
    print(f"  Log level: {args.log_level}")
    print(f"  Reload: {args.reload}")
    
    print(f"\nServer will be available at:")
    print(f"  API: http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/v1/health")
    
    # Start server
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 