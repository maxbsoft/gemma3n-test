# Quick Installation Guide for Gemma 3n API

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (RTX A4000 or similar)
- Hugging Face account with accepted Gemma license

## 🚀 Step-by-Step Installation

### 1. Preparation

```bash
# Navigate to server folder
cd server

# Run automatic installation
python setup.py
```

### 2. Token Setup

```bash
# Edit .env file in project root
nano ../.env

# Add your Hugging Face token:
HF_TOKEN=your_actual_token_here
```

### 3. Accept Gemma License

Go to https://huggingface.co/google/gemma-3n-e4b-it and accept the license.

### 4. Start Server

```bash
# Simple start
python start_server.py

# Or use bash script
./run.sh

# For development
./run.sh --dev
```

### 5. Verify Installation

```bash
# In new terminal
curl http://localhost:8000/v1/health

# Or run examples
python examples/basic_usage.py
```

## ⚡ Quick Commands

```bash
# Install and start (single command)
python setup.py && python start_server.py

# Start with quantized model (fast)
./run.sh --model gemma-3n-e2b-quantized

# Start with full E4B model (quality)
./run.sh --model gemma-3n-e4b-full

# Development with auto-reload
./run.sh --dev

# Custom port
./run.sh --port 8080
```

## 🐛 Troubleshooting

**Error "HF_TOKEN not found"**
```bash
echo "HF_TOKEN=hf_your_token_here" >> ../.env
```

**Error "CUDA out of memory"**
```bash
./run.sh --model gemma-3n-e2b-quantized
```

**Model won't load**
- Check internet connection
- Make sure you accepted Gemma license
- Verify token correctness

## 📚 What's Next?

1. **API documentation**: http://localhost:8000/docs
2. **Usage examples**: `examples/basic_usage.py`
3. **LangChain integration**: `examples/langchain_examples.py`
4. **Detailed documentation**: `README.md`

## 🆘 Support

- Check server logs
- Use `./run.sh --debug` for detailed logs
- Check status: `curl http://localhost:8000/v1/status` 