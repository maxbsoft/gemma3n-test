"""
LangChain integration for Gemma 3n API
"""
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import BaseMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
import requests
import asyncio
import aiohttp
import json
from pydantic import Field

class GemmaLLM(LLM):
    """
    LangChain LLM wrapper for Gemma 3n API (text only)
    """
    
    api_base: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="gemma-3n-e2b-quantized")
    temperature: float = Field(default=0.8)
    max_tokens: int = Field(default=150)
    top_p: float = Field(default=0.9)
    top_k: int = Field(default=50)
    preset: Optional[str] = Field(default=None)
    
    @property
    def _llm_type(self) -> str:
        return "gemma-3n"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make synchronous call to Gemma API"""
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": False
        }
        
        if self.preset:
            payload["preset"] = self.preset
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make asynchronous call to Gemma API"""
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": False
        }
        
        if self.preset:
            payload["preset"] = self.preset
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ValueError(f"API request failed: {response.status} - {text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]

class GemmaChatModel(BaseChatModel):
    """
    LangChain Chat Model wrapper for Gemma 3n API (supports multimodal)
    """
    
    api_base: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="gemma-3n-e2b-quantized")
    temperature: float = Field(default=0.8)
    max_tokens: int = Field(default=150)
    top_p: float = Field(default=0.9)
    top_k: int = Field(default=50)
    preset: Optional[str] = Field(default=None)
    streaming: bool = Field(default=False)
    
    @property
    def _llm_type(self) -> str:
        return "gemma-3n-chat"
    
    def _convert_messages_to_api_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to API format"""
        api_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # Default fallback
            
            api_messages.append({
                "role": role,
                "content": message.content
            })
        
        return api_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Generate response synchronously"""
        from langchain.schema import ChatGeneration, ChatResult
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": False
        }
        
        if self.preset:
            payload["preset"] = self.preset
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Generate response asynchronously"""
        from langchain.schema import ChatGeneration, ChatResult
        
        api_messages = self._convert_messages_to_api_format(messages)
        
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.streaming
        }
        
        if self.preset:
            payload["preset"] = self.preset
        
        async with aiohttp.ClientSession() as session:
            if self.streaming:
                # Handle streaming response
                content = ""
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise ValueError(f"API request failed: {response.status} - {text}")
                    
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if chunk["choices"][0]["delta"].get("content"):
                                    content += chunk["choices"][0]["delta"]["content"]
                            except:
                                continue
            else:
                # Handle regular response
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise ValueError(f"API request failed: {response.status} - {text}")
                    
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
        
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])

# Convenience functions
def create_gemma_llm(
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-3n-e2b-quantized",
    **kwargs
) -> GemmaLLM:
    """Create a Gemma LLM instance"""
    return GemmaLLM(api_base=api_base, model=model, **kwargs)

def create_gemma_chat_model(
    api_base: str = "http://localhost:8000/v1", 
    model: str = "gemma-3n-e2b-quantized",
    **kwargs
) -> GemmaChatModel:
    """Create a Gemma Chat Model instance"""
    return GemmaChatModel(api_base=api_base, model=model, **kwargs)

# Multimodal message helpers
def create_multimodal_message(text: str, image_url: Optional[str] = None, image_base64: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a multimodal message for direct API usage
    """
    content = [{"type": "text", "text": text}]
    
    if image_url:
        content.append({"type": "image", "url": image_url})
    elif image_base64:
        content.append({"type": "image", "base64": image_base64})
    
    return {
        "role": "user",
        "content": content
    }

async def call_multimodal_api(
    messages: List[Dict[str, Any]],
    api_base: str = "http://localhost:8000/v1",
    model: str = "gemma-3n-e2b-quantized",
    **kwargs
) -> str:
    """
    Direct multimodal API call for complex use cases
    """
    payload = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_base}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"API request failed: {response.status} - {text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"] 