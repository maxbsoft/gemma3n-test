"""
Pydantic schemas for OpenAI-compatible API
"""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Request schemas
class MessageContent(BaseModel):
    type: Literal["text", "image", "audio", "video"]
    text: Optional[str] = None
    url: Optional[str] = None
    base64: Optional[str] = None
    
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemma-3n-e2b-quantized", description="Model to use")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=150, le=2048)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=200)
    stream: Optional[bool] = False
    preset: Optional[Literal["creative", "balanced", "precise", "deterministic"]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "model": "gemma-3n-e2b-quantized",
                "messages": [
                    {"role": "user", "content": "Tell me about artificial intelligence"}
                ],
                "max_tokens": 150,
                "temperature": 0.8
            }
        }

# Response schemas
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "error"]] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = "gemma-3n-api-v1.0"

# Streaming response schemas
class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "error"]] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]

# Model management schemas
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "gemma-3n-api"
    
class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

class ModelLoadRequest(BaseModel):
    model: str = Field(..., description="Model name to load")
    
class ModelLoadResponse(BaseModel):
    success: bool
    message: str
    model_info: Optional[Dict[str, Any]] = None

# Health and status schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    gpu_available: bool
    gpu_memory: Optional[str] = None
    current_model: Optional[str] = None

class StatusResponse(BaseModel):
    server_status: str
    current_model: Optional[str] = None
    available_models: List[str]
    gpu_info: Optional[Dict[str, Any]] = None
    performance_stats: Optional[Dict[str, Any]] = None

# Error schemas
class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail 