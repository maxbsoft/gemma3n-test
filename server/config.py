"""
Configuration settings for Gemma 3n API Server
"""
import os
from typing import Dict, Any, List
from pydantic import BaseModel
from dotenv import load_dotenv
from pydantic.v1 import BaseSettings

load_dotenv()

class ModelConfig:
    """Configuration for different model variants"""
    
    # Quantized models (faster, less VRAM)
    QUANTIZED_MODELS = {
        "gemma-3n-e2b-quantized": {
            "model_id": "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
            "quantization": "4bit",
            "estimated_vram": "4-6GB",
            "expected_speed": "25-35 tokens/sec"
        }
    }
    
    # Full precision models (better quality, more VRAM)
    FULL_MODELS = {
        "gemma-3n-e2b-full": {
            "model_id": "google/gemma-3n-E2B-it",
            "torch_dtype": "bfloat16", 
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "quantization": None,
            "estimated_vram": "8-12GB",
            "expected_speed": "15-25 tokens/sec"
        },
        "gemma-3n-e4b-full": {
            "model_id": "google/gemma-3n-e4b-it",
            "torch_dtype": "bfloat16",
            "device_map": "auto", 
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "quantization": None,
            "estimated_vram": "8-12GB",
            "expected_speed": "15-25 tokens/sec"
        }
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available models"""
        return {**cls.QUANTIZED_MODELS, **cls.FULL_MODELS}
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        all_models = cls.get_all_models()
        if model_name not in all_models:
            raise ValueError(f"Model {model_name} not found. Available: {list(all_models.keys())}")
        return all_models[model_name]

class ServerConfig(BaseSettings):
    """Server configuration settings"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Model settings
    DEFAULT_MODEL: str = "gemma-3n-e2b-quantized"  # Start with quantized for faster loading
    MAX_CONCURRENT_REQUESTS: int = 4
    MODEL_TIMEOUT: int = 300  # 5 minutes
    
    # Generation defaults
    DEFAULT_MAX_TOKENS: int = 150
    DEFAULT_TEMPERATURE: float = 0.8
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 50
    
    # Multimodal settings
    MAX_IMAGE_SIZE: int = 2048  # pixels
    MAX_VIDEO_DURATION: int = 30  # seconds
    MAX_AUDIO_DURATION: int = 60  # seconds
    
    # Security and limits
    MAX_TOKENS_PER_REQUEST: int = 2048
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # HuggingFace token
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Model caching
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./model_cache")
    
    class Config:
        env_file = "../.env"

# Global configuration instance
config = ServerConfig()

# Generation parameters presets
GENERATION_PRESETS = {
    "creative": {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 100,
        "do_sample": True
    },
    "balanced": {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True
    },
    "precise": {
        "temperature": 0.3,
        "top_p": 0.5,
        "top_k": 20,
        "do_sample": True
    },
    "deterministic": {
        "do_sample": False
    }
} 