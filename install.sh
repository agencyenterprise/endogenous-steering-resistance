set -ex

apt-get update 
apt-get install -y ninja-build
uv venv --python 3.12
uv pip install sae_lens==6.13.0
uv pip install -r /workspace/vllm-sae/local_reqs/requirements.txt
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a2e6fa7e035ff058fc37fdaaf014707efff2fcf3/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl uv pip install -e /workspace/vllm-sae
uv pip install -r requirements.txt