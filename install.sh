#!/bin/bash
set -e

# Install dependencies for ESR experiments
# Assumes vllm-sae is cloned as a sibling directory to this repository

# Check if vllm-sae exists
if [ ! -d "../vllm-sae" ]; then
    echo "Error: vllm-sae not found at ../vllm-sae"
    echo "Please clone vllm-sae as a sibling directory:"
    echo "  cd .."
    echo "  git clone <vllm-sae-repo-url> vllm-sae"
    echo "  cd endogenous-steering-resistance"
    exit 1
fi

echo "Creating virtual environment..."
uv venv --python 3.12

echo "Installing SAE lens..."
uv pip install sae_lens==6.13.0

echo "Installing vLLM-SAE dependencies..."
uv pip install -r ../vllm-sae/local_reqs/requirements.txt

echo "Installing vLLM-SAE..."
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl uv pip install -e ../vllm-sae

echo "Installing experiment requirements..."
uv pip install -r requirements.txt

echo "Installation complete! Activate the environment with:"
echo "  source .venv/bin/activate"
