# Gemma 3n OpenAI-Compatible API Server

This project provides an OpenAI-compatible API for Google's Gemma 3n models, including multimodal support, model switching, and LangChain integration.

## 🚀 Quick Start

### Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

### Setup Hugging Face Token

Make sure your HF token is configured in `../.env` file:

```bash
# From project root directory
echo "HF_TOKEN=your_huggingface_token_here" >> .env
```

### Start Server

```bash
# Simple start
python start_server.py

# With additional options
python start_server.py --host 0.0.0.0 --port 8000 --model gemma-3n-e2b-quantized

# For development (with auto-reload)
python start_server.py --reload
```

Server will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs  
- Health check: http://localhost:8000/v1/health

## 📋 Features

### ✅ Supported Models

| Model | Size | VRAM | Speed | Description |
|-------|------|------|-------|-------------|
| `gemma-3n-e2b-quantized` | ~4GB | 4-6GB | 25-35 tok/s | Quantized model (default) |
| `gemma-3n-e2b-full` | ~8GB | 8-12GB | 15-25 tok/s | Full E2B model |
| `gemma-3n-e4b-full` | ~8GB | 8-12GB | 15-25 tok/s | Full E4B model |

### ✅ Supported Input Formats

- **Text** - regular text requests
- **Images** - image URLs or base64
- **Multimodal** - text + images in single request

### ✅ API Endpoints

- `POST /v1/chat/completions` - generate responses (OpenAI compatible)
- `GET /v1/models` - list available models
- `POST /v1/models/load` - load specific model
- `GET /v1/health` - server health check
- `GET /v1/status` - detailed server information

## 🛠 Usage Examples

### Basic Text Request

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gemma-3n-e2b-quantized",
    "messages": [
        {"role": "user", "content": "Tell me about quantum computers"}
    ],
    "max_tokens": 150,
    "temperature": 0.8
})

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Multimodal Request

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gemma-3n-e2b-quantized",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "url": "https://example.com/image.jpg"}
            ]
        }
    ],
    "max_tokens": 200
})

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Streaming Response

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", 
    json={
        "model": "gemma-3n-e2b-quantized", 
        "messages": [{"role": "user", "content": "Write a poem about AI"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 🔗 LangChain Integration

### Setup

```python
from integrations.langchain_integration import create_gemma_llm, create_gemma_chat_model
```

### Simple Usage

```python
# LLM for text tasks
llm = create_gemma_llm(
    model="gemma-3n-e2b-quantized",
    temperature=0.7,
    max_tokens=150
)

response = llm("Explain what machine learning is")
print(response)

# Chat model for conversations
from langchain.schema import HumanMessage, SystemMessage

chat_model = create_gemma_chat_model(model="gemma-3n-e2b-quantized")

messages = [
    SystemMessage(content="You are a helpful AI assistant"),
    HumanMessage(content="Tell me about neural networks")
]

response = chat_model(messages)
print(response.content)
```

### Chains and Memory

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = create_gemma_llm(model="gemma-3n-e2b-quantized")
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Conversation with memory
response1 = conversation.predict(input="Hi, I'm learning about AI")
response2 = conversation.predict(input="Tell me more about deep learning")
```

## ⚙️ Configuration

### Generation Parameters

```python
# Using presets
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gemma-3n-e2b-quantized",
    "messages": [{"role": "user", "content": "Write a story"}],
    "preset": "creative"  # creative, balanced, precise, deterministic
})

# Manual configuration
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gemma-3n-e2b-quantized",
    "messages": [{"role": "user", "content": "Explain the concept"}],
    "temperature": 0.3,    # Determinism (0.0-2.0)
    "top_p": 0.5,         # Nucleus sampling (0.0-1.0)
    "top_k": 20,          # Top-k sampling (1-200)
    "max_tokens": 200     # Maximum number of tokens
})
```

### Model Switching

```python
import requests

# Load E4B model
response = requests.post("http://localhost:8000/v1/models/load", json={
    "model": "gemma-3n-e4b-full"
})

if response.json()["success"]:
    print("E4B model loaded successfully")
    
    # Now use the new model
    response = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "gemma-3n-e4b-full",
        "messages": [{"role": "user", "content": "Hello!"}]
    })
```

## 🧪 Testing

### Run Examples

```bash
# Basic API examples
cd examples
python basic_usage.py

# LangChain examples
python langchain_examples.py
```

### Server Status Check

```bash
# Health check
curl http://localhost:8000/v1/health

# Detailed information
curl http://localhost:8000/v1/status

# List available models
curl http://localhost:8000/v1/models
```

## 🔧 Startup Parameters

```bash
python start_server.py --help

# Main options:
--host 0.0.0.0              # Host to bind to
--port 8000                 # Port
--model gemma-3n-e2b-quantized  # Default model
--workers 1                 # Number of workers  
--reload                    # Auto-reload for development
--log-level info           # Logging level
```

## 📊 Performance

### RTX A4000 (16GB)

| Model | Loading | Memory | Speed |
|-------|---------|--------|-------|
| Quantized E2B | 30-45s | 4-6GB | 25-35 tok/s |
| Full E2B | 60-90s | 8-10GB | 15-25 tok/s |
| Full E4B | 60-90s | 8-10GB | 15-25 tok/s |

### Optimization Recommendations

1. **For maximum speed**: use quantized model
2. **For better quality**: use full model
3. **For memory saving**: unload unused models
4. **For production**: use multiple workers

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use quantized model
python start_server.py --model gemma-3n-e2b-quantized

# Or clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

**Slow Generation**
```bash
# Check GPU usage
curl http://localhost:8000/v1/status
# gpu_available should be true

# Check current model
curl http://localhost:8000/v1/models/current
```

**Authorization Errors**
```bash
# Check HF token
echo $HF_TOKEN

# Make sure Gemma license is accepted
# https://huggingface.co/google/gemma-3n-e4b-it
```

### Logs and Debugging

```bash
# Run with detailed logs
python start_server.py --log-level debug

# Check status
tail -f server.log
```

## 🚀 Deployment

### Production Server

```bash
# Install Gunicorn for production
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (optional)

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_server.py", "--host", "0.0.0.0"]
```

## 📝 API Documentation

After starting the server, documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Create Pull Request

## 📄 License

This project uses Gemma models which require accepting Google's license.

## 🆘 Support

- GitHub Issues for bugs and feature requests
- API documentation in Swagger UI
- Examples in `examples/` folder

---

**Happy coding with Gemma 3n API! 🎉** 