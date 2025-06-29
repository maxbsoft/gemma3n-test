# Gemma 3n Testing Suite

This project provides comprehensive testing and examples for Google's Gemma 3n multimodal models, including performance benchmarks, sampling parameter tests, and pipeline examples.

## Project Structure

```
gemma_test/
├── main.py                       # Main performance testing suite
├── test_sampling_fixed.py        # Sampling parameters testing
├── gemma3n_pipeline_example.py   # Simple pipeline example
├── requirements.txt              # Python dependencies
├── setup_environment.sh          # Environment setup script
├── .env                         # Environment variables (create this)
└── README.md                    # This file
```

## Features

### 1. Main Performance Tests (`main.py`)
- **English Text Generation** - Creative story generation with speed metrics
- **Ukrainian Text Generation** - Multilingual capability testing  
- **Image Processing** - Multimodal image description
- **Performance Comparison** - Greedy vs sampling generation
- Uses quantized model: `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit`

### 2. Sampling Parameter Tests (`test_sampling_fixed.py`)
- Tests different `top_k`, `top_p`, `temperature` combinations
- Compares deterministic vs stochastic generation
- Performance analysis for each parameter set
- Uses original model: `google/gemma-3n-e4b-it`

### 3. Pipeline Example (`gemma3n_pipeline_example.py`)
- Simple multimodal pipeline usage
- Image + text input examples
- Text-only generation examples
- Easy-to-understand implementation

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

## Running the Tests

### Main Performance Suite
```bash
python main.py
```
**Features:**
- Comprehensive performance testing
- Multiple generation strategies comparison
- Memory usage monitoring
- Speed benchmarking (tokens/second)

### Sampling Parameters Testing
```bash
python test_sampling_fixed.py
```
**Features:**
- Tests 7 different parameter combinations
- Temperature effects analysis
- Creativity vs coherence comparison
- Detailed parameter impact analysis

### Simple Pipeline Example
```bash
python gemma3n_pipeline_example.py
```
**Features:**
- Basic multimodal usage
- Image description from URL
- Text-only generation
- Beginner-friendly code

## Expected Performance

### RTX A4000 (16GB)
- **Main model (quantized)**: 15-25 tokens/sec
- **Full model**: 10-20 tokens/sec
- **Memory usage**: 4-8GB
- **Loading time**: 30-60 seconds

### RTX 4090 (24GB)
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

```
transformers>=4.53.0    # Gemma 3n support
torch>=2.0.0           # GPU acceleration
bitsandbytes           # Quantization
accelerate             # Model loading optimization
Pillow                 # Image processing
python-dotenv          # Environment variables
librosa                # Audio processing (future use)
```

## Generated Files

During testing, these files may be created:
- `test_image.png` - Test pattern image for multimodal tests

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

## Contributing

To add new tests or improve existing ones:
1. Follow the existing code structure
2. Add appropriate error handling
3. Include performance measurements
4. Update this README if needed

## License

This testing suite is provided for educational and research purposes. Please ensure you comply with Google's Gemma license terms when using the models. 