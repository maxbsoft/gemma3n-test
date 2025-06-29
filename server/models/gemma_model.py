"""
Gemma 3n Model Handler with multimodal support
"""
import os
import time
import asyncio
import threading
import queue
import torch
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoProcessor as ProcessorType
    from transformers.modeling_utils import PreTrainedModel as ModelType
else:
    ProcessorType = Any
    ModelType = Any
from transformers import AutoProcessor, Gemma3nForConditionalGeneration  # type: ignore
from PIL import Image
import io
import base64
import requests
import logging
import sys
from pathlib import Path
import cv2
import numpy as np
import tempfile
sys.path.append(str(Path(__file__).parent.parent))

from config import ModelConfig, config, GENERATION_PRESETS

logger = logging.getLogger(__name__)

class GemmaModelHandler:
    """
    Handler for Gemma 3n models with OpenAI-compatible interface
    """
    
    def __init__(self):
        self.model: Optional[ModelType] = None
        self.processor: Optional[ProcessorType] = None
        self.current_model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = None
        
        # Optimization settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
    async def load_model(self, model_name: str) -> bool:
        """
        Load specified model asynchronously
        """
        if self.current_model_name == model_name and self.model is not None:
            logger.info(f"Model {model_name} already loaded")
            return True
            
        try:
            logger.info(f"Loading model: {model_name}")
            self.model_config = ModelConfig.get_model_config(model_name)
            
            if not config.HF_TOKEN:
                raise ValueError("HF_TOKEN not found in environment variables")
            
            start_time = time.time()
            
            # Load model in separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model, self.processor = await loop.run_in_executor(
                None, self._load_model_sync, self.model_config
            )
            
            load_time = time.time() - start_time
            self.current_model_name = model_name
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
                logger.info(f"GPU Memory allocated: {memory_allocated:.2f} GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _load_model_sync(self, model_config: Dict[str, Any]):
        """Synchronous model loading"""
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_config["model_id"],
            device_map=model_config.get("device_map", "auto"),
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
            token=config.HF_TOKEN,
            low_cpu_mem_usage=model_config.get("low_cpu_mem_usage", True),
            trust_remote_code=model_config.get("trust_remote_code", False)
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            model_config["model_id"], 
            token=config.HF_TOKEN
        )
        
        return model, processor
    
    def unload_model(self):
        """Unload current model to free memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_name = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
            
            logger.info("Model unloaded and memory cleared")
    
    def force_cleanup_gpu_memory(self):
        """Force cleanup of GPU memory"""
        logger.warning("Force cleaning GPU memory...")
        
        if torch.cuda.is_available():
            # Clear all caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Clean up shared memory
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            gpu_memory_after = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            gpu_memory_after_gb = gpu_memory_after / (1024**3)
            logger.info(f"GPU memory after cleanup: {gpu_memory_after_gb:.2f} GB available")
    
    def _process_image_content(self, content: Dict[str, Any]) -> Image.Image:
        """Process image from various sources"""
        if "url" in content:
            # Load from URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(content["url"], headers=headers, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        elif "image" in content:
            # Direct PIL Image
            image = content["image"]
        elif "base64" in content:
            # Base64 encoded image
            image_data = base64.b64decode(content["base64"])
            image = Image.open(io.BytesIO(image_data))
        else:
            raise ValueError("Image content must have 'url', 'image', or 'base64' field")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_size = config.MAX_IMAGE_SIZE
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
    
    def _process_audio_content(self, content: Dict[str, Any]) -> str:
        """Process audio from various sources"""
        if "audio" in content:
            # Direct audio URL (Gemma format)
            return content["audio"]
        elif "url" in content:
            # Audio URL in url field
            return content["url"]
        else:
            raise ValueError("Audio content must have 'audio' or 'url' field")
    
    def _process_video_content(self, content: Dict[str, Any]) -> List[Image.Image]:
        """Process video from various sources and extract frames"""
        if "url" in content:
            video_url = content["url"]
            logger.info(f"Processing video URL: {video_url}")
            
            # Download video to temporary file
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(video_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            try:
                # Extract frames using OpenCV
                frames = self._extract_video_frames(temp_path)
                logger.info(f"Extracted {len(frames)} frames from video")
                return frames
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            raise ValueError("Video content must have 'url' field")
    
    def _extract_video_frames(self, video_path: str, max_frames: int = 10) -> List[Image.Image]:
        """Extract frames from video file"""
        frames = []
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
            
            # Calculate frame interval to get evenly distributed frames
            if total_frames <= max_frames:
                frame_interval = 1
            else:
                frame_interval = total_frames // max_frames
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Take every nth frame
                if frame_count % frame_interval == 0 and len(frames) < max_frames:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Resize if too large
                    max_size = config.MAX_IMAGE_SIZE
                    if max(pil_image.size) > max_size:
                        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    frames.append(pil_image)
                
                frame_count += 1
            
        finally:
            cap.release()
        
        return frames
    
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Gemma format"""
        processed_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, str):
                # Simple text message
                processed_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}]
                })
            elif isinstance(content, list):
                # Multi-part content (text + images)
                processed_content = []
                
                for item in content:
                    if item["type"] == "text":
                        processed_content.append(item)
                    elif item["type"] == "image":
                        # Process image
                        image = self._process_image_content(item)
                        processed_content.append({"type": "image", "image": image})
                    elif item["type"] == "audio":
                        # Process audio
                        audio_url = self._process_audio_content(item)
                        processed_content.append({"type": "audio", "audio": audio_url})
                    elif item["type"] == "video":
                        # Check GPU memory before processing video
                        if torch.cuda.is_available():
                            gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                            gpu_memory_free_gb = gpu_memory_free / (1024**3)
                            logger.info(f"Free GPU memory before video processing: {gpu_memory_free_gb:.2f} GB")
                            
                            if gpu_memory_free_gb < 2.0:  # Need at least 2GB free
                                logger.warning("Low GPU memory, clearing cache before video processing")
                                torch.cuda.empty_cache()
                        
                        # Process video - extract frames and add as images
                        video_frames = self._process_video_content(item)
                        logger.info(f"Adding {len(video_frames)} video frames as images")
                        
                        # Add each frame as an image
                        for i, frame in enumerate(video_frames):
                            frame_content = {"type": "image", "image": frame}
                            processed_content.append(frame_content)
                        
                        # Clear GPU cache after processing video
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.info("Cleared GPU cache after video processing")
                
                processed_messages.append({
                    "role": role,
                    "content": processed_content
                })
        
        logger.info(f"Final processed messages: {processed_messages}")
        return processed_messages
    
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        preset: Optional[str] = None,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate response with OpenAI-compatible parameters
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Check GPU memory at start
        if torch.cuda.is_available():
            gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            gpu_memory_free_gb = gpu_memory_free / (1024**3)
            logger.info(f"Available GPU memory: {gpu_memory_free_gb:.2f} GB")
            
            if gpu_memory_free_gb < 1.0:  # Critical memory low
                logger.error(f"GPU memory critically low: {gpu_memory_free_gb:.2f} GB available")
                torch.cuda.empty_cache()
                
                # Recheck after cache clear
                gpu_memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                gpu_memory_free_gb = gpu_memory_free / (1024**3)
                
                if gpu_memory_free_gb < 0.5:
                    raise RuntimeError(f"Insufficient GPU memory: {gpu_memory_free_gb:.2f} GB available, need at least 0.5 GB")
        
        # Set default parameters
        max_tokens = max_tokens or config.DEFAULT_MAX_TOKENS
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else config.DEFAULT_TOP_P
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        
        # Apply preset if specified
        if preset and preset in GENERATION_PRESETS:
            preset_params = GENERATION_PRESETS[preset]
            temperature = preset_params.get("temperature", temperature)
            top_p = preset_params.get("top_p", top_p)
            top_k = preset_params.get("top_k", top_k)
        
        # Prepare messages
        processed_messages = self._prepare_messages(messages)
        
        # Tokenize input
        inputs = self.processor.apply_chat_template(  # type: ignore
            processed_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        if stream:
            return self._generate_stream(inputs, input_len, max_tokens, temperature, top_p, top_k, **kwargs)
        else:
            return await self._generate_complete(inputs, input_len, max_tokens, temperature, top_p, top_k, **kwargs)
    
    async def _generate_complete(
        self, inputs, input_len, max_tokens, temperature, top_p, top_k, **kwargs
    ) -> str:
        """Generate complete response"""
        start_time = time.time()
        
        with torch.inference_mode():
            generation = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.processor.tokenizer.eos_token_id,  # type: ignore
                use_cache=True,
                **kwargs
            )
            
            generation = generation[0][input_len:]
        
        generation_time = time.time() - start_time
        decoded = self.processor.decode(generation, skip_special_tokens=True)  # type: ignore
        
        # Log performance metrics
        tokens_per_second = len(generation) / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {len(generation)} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return decoded
    
    async def _generate_stream(
        self, inputs, input_len, max_tokens, temperature, top_p, top_k, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response token by token"""
        import threading
        import queue
        
        # Queue for streaming tokens
        token_queue = queue.Queue()
        
        def generation_thread():
            """Run generation in separate thread"""
            try:
                with torch.inference_mode():
                    # Use TextIteratorStreamer for real streaming
                    from transformers.generation.streamers import TextIteratorStreamer
                    
                    streamer = TextIteratorStreamer(
                        self.processor.tokenizer,  # type: ignore
                        skip_prompt=True,
                        skip_special_tokens=True,
                        timeout=30.0
                    )
                    
                    generation_kwargs = {
                        **inputs,
                        "max_new_tokens": max_tokens,
                        "do_sample": True,
                        "top_k": top_k,
                        "top_p": top_p,
                        "temperature": temperature,
                        "pad_token_id": self.processor.tokenizer.eos_token_id,  # type: ignore
                        "use_cache": True,
                        "streamer": streamer,
                        **kwargs
                    }
                    
                    # Start generation in background
                    thread = threading.Thread(
                        target=self.model.generate,  # type: ignore
                        kwargs=generation_kwargs
                    )
                    thread.start()
                    
                    # Stream tokens as they're generated
                    for new_text in streamer:
                        if new_text:
                            token_queue.put(new_text)
                    
                    # Signal end of generation
                    token_queue.put(None)
                    thread.join()
                    
            except Exception as e:
                logger.error(f"Error in streaming generation: {e}")
                token_queue.put(None)
        
        # Start generation thread
        gen_thread = threading.Thread(target=generation_thread)
        gen_thread.start()
        
        # Yield tokens as they arrive
        try:
            while True:
                # Wait for next token with timeout
                try:
                    token = token_queue.get(timeout=1.0)
                    if token is None:  # End of generation
                        break
                    yield token
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                except queue.Empty:
                    # Continue waiting
                    continue
        finally:
            # Ensure thread cleanup
            if gen_thread.is_alive():
                gen_thread.join(timeout=5.0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        if self.current_model_name is None:
            return {"status": "no_model_loaded"}
        
        info = {
            "model_name": self.current_model_name,
            "model_id": self.model_config["model_id"],  # type: ignore
            "quantization": self.model_config.get("quantization"),  # type: ignore
            "estimated_vram": self.model_config.get("estimated_vram"),  # type: ignore
            "expected_speed": self.model_config.get("expected_speed"),  # type: ignore
            "device": str(self.device)
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name()
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        
        return info

# Global model handler instance
model_handler = GemmaModelHandler() 