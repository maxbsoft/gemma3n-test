# Gemma 3n Testing Suite

This project provides comprehensive testing and examples for Google's Gemma 3n multimodal models, including an OpenAI-compatible API server, performance benchmarks, sampling parameter tests, and integration examples.

## Project Structure

```
gemma_test/
├── server/                       # Production-ready API server
│   ├── main.py                   # FastAPI server entry point
│   ├── config.py                 # Configuration and model settings
│   ├── schemas.py                # Pydantic models for API
│   ├── models/
│   │   └── gemma_model.py        # Model handler with multimodal support
│   ├── api/
│   │   └── endpoints.py          # OpenAI-compatible endpoints
│   ├── integrations/
│   │   └── langchain_integration.py  # LangChain compatibility
│   ├── examples/
│   │   ├── basic_usage.py        # API usage examples
│   │   └── langchain_examples.py # LangChain integration examples
│   ├── setup.py                 # Automatic installation script
│   ├── start_server.py          # Server startup with CLI options
│   ├── run.sh                   # Bash script for server management
│   ├── README.md                # Server documentation
│   └── INSTALL.md               # Quick installation guide
├── main.py                       # Performance testing suite
├── test_sampling_fixed.py        # Sampling parameters testing
├── gemma3n_pipeline_example.py   # Simple pipeline example
├── requirements.txt              # Python dependencies
├── setup_environment.sh          # Environment setup script
├── .env                         # Environment variables (create this)
└── README.md                    # This file
```

## Features

### 🚀 Production API Server (`server/`)
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **Multimodal Support** - Text, images, audio, and video processing
- **Model Switching** - Dynamic loading of quantized and full models
- **Streaming Responses** - Real-time token generation
- **LangChain Integration** - Ready-to-use LLM and ChatModel classes
- **Performance Optimized** - CUDA optimization for RTX A4000/4090
- **Generation Presets** - Creative, balanced, precise, and deterministic modes

### 🧪 Performance Testing Suite
#### 1. Main Performance Tests (`main.py`)
- **English Text Generation** - Creative story generation with speed metrics
- **Ukrainian Text Generation** - Multilingual capability testing  
- **Image Processing** - Multimodal image description
- **Performance Comparison** - Greedy vs sampling generation
- Uses quantized model: `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit`

#### 2. Sampling Parameter Tests (`test_sampling_fixed.py`)
- Tests different `top_k`, `top_p`, `temperature` combinations
- Compares deterministic vs stochastic generation
- Performance analysis for each parameter set
- Uses original model: `google/gemma-3n-e4b-it`

#### 3. Pipeline Example (`gemma3n_pipeline_example.py`)
- Simple multimodal pipeline usage
- Image + text input examples
- Text-only generation examples
- Easy-to-understand implementation

## Quick Start (API Server)

### 🚀 One-Command Setup
```bash
cd server
python setup.py
```
This automatically installs dependencies, sets up the environment, and starts the server.

### 🔥 Manual Setup
```bash
cd server
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
python main.py
```

### 💬 Test the API
```bash
# Test basic chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3n-e2b-quantized",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Test with image
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3n-e2b-quantized", 
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image", "url": "https://example.com/image.jpg"}
      ]
    }]
  }'
```

See detailed server documentation: [`server/README.md`](server/README.md)

> **⚠️ Production Warning**: Server has a known blocking issue - it can only process one request at a time. For production use, implement multi-instance setup (see "Known Issues" section).

---

## Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (RTX A4000/RTX 4090 recommended)
- **Hugging Face account** with Gemma license accepted
- **8GB+ GPU memory** (for full model) or 4GB+ (for quantized model)

## Setup Instructions

### 1. Accept Gemma License
Visit [Gemma 3n on Hugging Face](https://huggingface.co/google/gemma-3n-e4b-it) and accept the license.

### 2. Get Hugging Face Token
1. Go to [HF Settings > Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token for step 4

### 3. Clone and Setup Environment
```bash
# Clone the repository (or download files)
cd gemma_test

# Run the setup script
chmod +x setup_environment.sh
./setup_environment.sh
```

### 4. Configure Environment Variables
Edit the `.env` file and add your token:
```bash
# Edit .env file
nano .env

# Add your token:
HF_TOKEN=your_actual_huggingface_token_here
```

### 5. Activate Environment
```bash
source venv/bin/activate
```

## Usage Options

### 🚀 Production API Server
```bash
cd server
python main.py
# Server runs on http://localhost:8000
```
**API Endpoints:**
- `POST /v1/chat/completions` - Main chat endpoint (OpenAI compatible)
- `GET /v1/models` - List available models
- `POST /v1/models/load` - Switch between models
- `GET /v1/health` - Health check
- `GET /v1/status` - Detailed server status

**LangChain Integration:**
```python
from server.integrations.langchain_integration import GemmaChatModel

llm = GemmaChatModel(api_base="http://localhost:8000/v1")
response = llm.invoke("Hello, how are you?")
```

### 🧪 Performance Testing Suite

#### Main Performance Tests
```bash
python main.py
```
**Features:**
- Comprehensive performance testing
- Multiple generation strategies comparison
- Memory usage monitoring
- Speed benchmarking (tokens/second)

#### Sampling Parameters Testing
```bash
python test_sampling_fixed.py
```
**Features:**
- Tests 7 different parameter combinations
- Temperature effects analysis
- Creativity vs coherence comparison
- Detailed parameter impact analysis

#### Simple Pipeline Example
```bash
python gemma3n_pipeline_example.py
```
**Features:**
- Basic multimodal usage
- Image description from URL
- Text-only generation
- Beginner-friendly code

## Expected Performance

> **📊 Performance Disclaimer**: The performance figures below are approximate and may vary significantly based on hardware configuration, model size, prompt complexity, system load, and other factors. Use these numbers as rough estimates for planning purposes.

### 🚀 API Server Performance
#### RTX A4000 (16GB)
- **Quantized models**: 25-35 tokens/sec
- **Full models**: 15-25 tokens/sec
- **Memory usage**: 4-6GB (quantized), 8-12GB (full)
- **Cold start**: 30-60 seconds
- **Concurrent requests**: ⚠️ **1 only** (blocking issue)

#### RTX 4090 (24GB)
- **Quantized models**: 35-50 tokens/sec
- **Full models**: 25-35 tokens/sec
- **Memory usage**: 6-8GB (quantized), 12-16GB (full)
- **Cold start**: 20-40 seconds
- **Concurrent requests**: ⚠️ **1 only** (blocking issue)

> **⚠️ Important**: Due to unresolved blocking issue, server can only process one request at a time. For production deployment, multi-instance setup with load balancer is recommended.

### 🧪 Testing Suite Performance
#### RTX A4000 (16GB)
- **Main model (quantized)**: 15-25 tokens/sec
- **Full model**: 10-20 tokens/sec
- **Memory usage**: 4-8GB
- **Loading time**: 30-60 seconds

#### RTX 4090 (24GB)
- **Main model (quantized)**: 25-35 tokens/sec  
- **Full model**: 20-30 tokens/sec
- **Memory usage**: 6-12GB
- **Loading time**: 20-40 seconds

## Model Differences

| File | Model Used | Size | Purpose |
|------|------------|------|---------|
| `main.py` | `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit` | ~4GB | Optimized performance testing |
| `test_sampling_fixed.py` | `google/gemma-3n-e4b-it` | ~8GB | Parameter analysis |
| `gemma3n_pipeline_example.py` | `google/gemma-3n-E4B-it` | ~8GB | Pipeline demonstration |

## Key Dependencies

### Core Dependencies
```
transformers>=4.53.0    # Gemma 3n support
torch>=2.0.0           # GPU acceleration
bitsandbytes           # Quantization
accelerate             # Model loading optimization
Pillow                 # Image processing
python-dotenv          # Environment variables
```

### Server Dependencies
```
fastapi>=0.104.0       # Web framework
uvicorn>=0.24.0        # ASGI server
pydantic>=2.5.0        # Data validation
sse-starlette>=1.8.0   # Server-sent events for streaming
opencv-python>=4.8.0   # Video processing
requests>=2.31.0       # HTTP client
```

### LangChain Integration
```
langchain>=0.1.0       # LLM framework
langchain-community    # Community integrations
aiohttp>=3.9.0         # Async HTTP client
```

## Generated Files

During testing, these files may be created:
- `test_image.png` - Test pattern image for multimodal tests

## Known Issues & Limitations

### 🚨 Critical Production Issues

**1. Request Blocking During Generation (UNRESOLVED)**
```
⚠️  IMPORTANT: Server blocks during response generation!

Issue: Server cannot accept new requests while processing current request
until text generation is completed.

Symptoms:
- New requests hang in waiting state
- Client-side timeouts
- Unable to handle concurrent requests
- Server appears "frozen" during long generations

Temporary Solutions:
- Use short max_tokens (<100)
- Avoid simultaneous requests
- Consider running multiple server instances on different ports
```

**2. Memory Leaks During Long Sessions**
```bash
# Model may accumulate memory during long operations
# Periodically restart the server:

# Automatic restart every 4 hours
while true; do
  timeout 14400 python main.py || echo "Restarting server..."
  sleep 5
done
```

**3. Streaming Responses Issues**
```
- Streaming may interrupt with large responses
- Client timeout during slow generation
- Uneven token speed in streaming mode
```

### 🔧 Workarounds

**Multi-Instance Setup (Recommended for Production)**
```bash
# Run multiple instances on different ports
cd server

# Instance 1
PORT=8000 python main.py &

# Instance 2  
PORT=8001 python main.py &

# Instance 3
PORT=8002 python main.py &

# Use nginx for load balancing
```

**Nginx Load Balancer Config**
```nginx
upstream gemma_servers {
    server localhost:8000;
    server localhost:8001; 
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://gemma_servers;
        proxy_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

**Client-Side Timeout Settings**
```python
# Increase timeout on client side
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload,
    timeout=300  # 5 minutes
)
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce max_new_tokens in the scripts
# Or use the quantized model version
```

**2. Model Download Fails**
```bash
# Check your HF_TOKEN in .env
# Ensure you accepted the Gemma license
# Check internet connection
```

**3. Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

**4. Pipeline Errors**
```bash
# Update transformers to latest version
pip install --upgrade transformers

# Or install development version
pip install git+https://github.com/huggingface/transformers.git
```

## Monitoring GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or install nvtop for better interface
sudo apt install nvtop
nvtop

# Python GPU monitoring
pip install gpustat
gpustat -i 1
```

## Use Cases

### 🤖 AI Applications
- **Chatbots and Virtual Assistants** - Multimodal conversation AI
- **Content Analysis** - Process images, videos, and audio
- **Educational Tools** - Interactive learning with multimodal input
- **Creative Applications** - Generate stories, descriptions, and more

### 🔬 Research and Development  
- **Model Performance Analysis** - Comprehensive benchmarking
- **Parameter Optimization** - Fine-tune generation settings
- **Multimodal Research** - Explore cross-modal understanding
- **Integration Testing** - Validate LangChain and API compatibility

### 🏢 Enterprise Integration
- **OpenAI API Replacement** - Drop-in replacement for existing systems ⚠️
- **On-Premise Deployment** - Keep data secure with local processing
- **Custom Model Switching** - Dynamic model selection based on requirements
- **Multi-Instance Architecture** - Scale with multiple server instances (workaround required)

> **Note**: Due to request blocking issues, enterprise deployment requires load balancer setup with multiple server instances.

## Advanced Features

### Model Management
```bash
# Switch models via API
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma-3n-e4b-full"}'

# Check server status
curl http://localhost:8000/v1/status
```

### Generation Presets
```python
# Use different generation modes
response = await model_handler.generate_response(
    messages=messages,
    preset="creative"  # Options: creative, balanced, precise, deterministic
)
```

### Streaming with LangChain
```python
from server.integrations.langchain_integration import GemmaChatModel

llm = GemmaChatModel(streaming=True)
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## Contributing

To add new features or improve existing ones:
1. Follow the existing code structure
2. Add appropriate error handling
3. Include performance measurements
4. Add tests and examples
5. Update documentation

## License

This testing suite is provided for educational and research purposes. Please ensure you comply with Google's Gemma license terms when using the models. 