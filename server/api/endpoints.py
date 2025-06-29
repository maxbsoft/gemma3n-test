"""
FastAPI endpoints for OpenAI-compatible Gemma 3n API
"""
import asyncio
import logging
import torch
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse,
    ModelLoadRequest, ModelLoadResponse, ModelsResponse, ModelInfo,
    HealthResponse, StatusResponse, ErrorResponse, Choice, Usage,
    StreamChoice, DeltaMessage
)
from models.gemma_model import model_handler
from config import ModelConfig, config, GENERATION_PRESETS

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI compatible)
    """
    try:
        # Ensure model is loaded
        if model_handler.current_model_name != request.model:
            success = await model_handler.load_model(request.model)
            if not success:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to load model {request.model}"
                )
        
        # Generate response
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            response_text = await model_handler.generate_response(
                messages=[msg.dict() for msg in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                preset=request.preset,
                stream=False
            )
            
            # Create OpenAI-compatible response
            choice = Choice(
                message={
                    "role": "assistant",  # type: ignore
                    "content": response_text
                }
            )
            
            return ChatCompletionResponse(
                model=request.model,
                choices=[choice],
                usage=Usage(completion_tokens=len(response_text.split()))  # type: ignore
            )
            
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _stream_chat_completion(request: ChatCompletionRequest):
    """Generate streaming chat completion"""
    try:
        response_generator = await model_handler.generate_response(
            messages=[msg.dict() for msg in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            preset=request.preset,
            stream=True
        )
        
        # Create base response ID and timestamp
        import uuid
        from datetime import datetime
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(datetime.now().timestamp())
        
        async for chunk in response_generator:  # type: ignore
            stream_response = ChatCompletionStreamResponse(
                id=response_id,
                created=created,
                model=request.model,
                choices=[StreamChoice(
                    delta=DeltaMessage(content=chunk)
                )]
            )
            
            yield f"data: {stream_response.json()}\n\n"
        
        # Send final chunk
        final_response = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=request.model,
            choices=[StreamChoice(
                delta=DeltaMessage(),
                finish_reason="stop"
            )]
        )
        yield f"data: {final_response.json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming completion: {e}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {error_response}\n\n"

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI compatible)"""
    all_models = ModelConfig.get_all_models()
    
    model_list = [
        ModelInfo(
            id=model_name,
            owned_by="gemma-3n-api"
        )
        for model_name in all_models.keys()
    ]
    
    return ModelsResponse(data=model_list)

@router.post("/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest):
    """Load a specific model"""
    try:
        success = await model_handler.load_model(request.model)
        if success:
            model_info = model_handler.get_model_info()
            return ModelLoadResponse(
                success=True,
                message=f"Model {request.model} loaded successfully",
                model_info=model_info
            )
        else:
            return ModelLoadResponse(
                success=False,
                message=f"Failed to load model {request.model}"
            )
    except Exception as e:
        logger.error(f"Error loading model {request.model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/unload")
async def unload_model():
    """Unload current model to free memory"""
    try:
        model_handler.unload_model()
        return {"success": True, "message": "Model unloaded successfully"}
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gpu/cleanup")
async def cleanup_gpu_memory():
    """Force cleanup of GPU memory"""
    try:
        model_handler.force_cleanup_gpu_memory()
        
        # Get memory status after cleanup
        if torch.cuda.is_available():
            gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            gpu_memory_free_gb = gpu_memory_free / (1024**3)
            return {
                "success": True, 
                "message": "GPU memory cleaned up successfully",
                "free_memory_gb": round(gpu_memory_free_gb, 2)
            }
        else:
            return {"success": True, "message": "No GPU available"}
    except Exception as e:
        logger.error(f"Error cleaning up GPU memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/current")
async def get_current_model():
    """Get information about currently loaded model"""
    model_info = model_handler.get_model_info()
    return model_info

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available:
        gpu_memory = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated"
    
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_memory=gpu_memory,
        current_model=model_handler.current_model_name
    )

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get detailed server status"""
    import torch
    import psutil
    
    # GPU info
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        }
    
    # Performance stats
    performance_stats = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
    
    return StatusResponse(
        server_status="running",
        current_model=model_handler.current_model_name,
        available_models=list(ModelConfig.get_all_models().keys()),
        gpu_info=gpu_info,
        performance_stats=performance_stats
    )

# Additional utility endpoints
@router.get("/presets")
async def get_generation_presets():
    """Get available generation presets"""
    return {"presets": GENERATION_PRESETS}

@router.get("/config")
async def get_server_config():
    """Get server configuration"""
    return {
        "default_model": config.DEFAULT_MODEL,
        "max_tokens_per_request": config.MAX_TOKENS_PER_REQUEST,
        "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
        "supported_modalities": ["text", "image", "video", "audio"],
        "gpu_available": torch.cuda.is_available() 
    } 