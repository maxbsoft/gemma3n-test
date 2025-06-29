#!/bin/bash

# Gemma 3n Testing Environment Setup Script
echo "Setting up Gemma 3n testing environment..."

# Create virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env template file..."
    echo "HF_TOKEN=your_huggingface_token_here" > .env
    echo "Please edit .env file and add your Hugging Face token!"
fi

echo "Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "source venv/bin/activate"
echo ""
echo "To run tests:"
echo "python main.py"
echo ""
echo "Don't forget to add your HF_TOKEN to the .env file!" 