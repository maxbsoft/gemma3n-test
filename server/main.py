"""
Gemma 3n OpenAI-Compatible API Server
"""
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import torch

from config import config, ModelConfig
from models.gemma_model import model_handler
from api.endpoints import router
from schemas import ErrorResponse, ErrorDetail

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Gemma 3n API Server...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load default model
    logger.info(f"Loading default model: {config.DEFAULT_MODEL}")
    success = await model_handler.load_model(config.DEFAULT_MODEL)
    if success:
        logger.info("Default model loaded successfully")
    else:
        logger.warning("Failed to load default model")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down...")
    model_handler.unload_model()
    logger.info("Server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Gemma 3n API",
    description="OpenAI-compatible API for Gemma 3n multimodal models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.detail,
                type="http_error",
                code=str(exc.status_code)
            )
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message="Internal server error",
                type="server_error",
                code="500"
            )
        ).dict()
    )

# Include API routes
app.include_router(router, prefix="/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gemma 3n OpenAI-Compatible API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/v1/health",
            "status": "/v1/status"
        },
        "features": [
            "Multimodal support (text + images)",
            "Model switching",
            "Streaming responses",
            "Generation presets",
            "OpenAI compatibility"
        ]
    }

# Additional info endpoint
@app.get("/info")
async def get_info():
    """Get detailed API information"""
    available_models = ModelConfig.get_all_models()
    
    return {
        "server": {
            "name": "Gemma 3n API Server",
            "version": "1.0.0",
            "gpu_available": torch.cuda.is_available(),
            "current_model": model_handler.current_model_name
        },
        "models": {
            "available": list(available_models.keys()),
            "default": config.DEFAULT_MODEL,
            "configurations": available_models
        },
        "limits": {
            "max_tokens_per_request": config.MAX_TOKENS_PER_REQUEST,
            "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
            "max_image_size": config.MAX_IMAGE_SIZE
        },
        "supported_features": [
            "Text generation",
            "Image understanding", 
            "Multimodal conversations",
            "Streaming responses",
            "Model hot-swapping"
        ]
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info"
    ) 