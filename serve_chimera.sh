#!/bin/bash

# Usage: ./serve_vllm.sh /path/to/model_dir
# If no argument is provided, it defaults to the test path.

MODEL_DIR=${1:-"/nvme_zone3/home/ekamai1/chimera/data/exports/zoro2_v2_full"}
TRANSFORMERS_DIR="/nvme_zone3/home/ekamai1/chimera/repos/transformers"

echo "Serving vLLM with model from: $MODEL_DIR"
echo "Using transformers from: $TRANSFORMERS_DIR"

# Clean up existing container
docker rm -f chimera-vllm 2>/dev/null

docker run -d \
  --name chimera-vllm \
  --gpus '"device=4,5,6,7"' \
  --ipc=host \
  -p 127.0.0.1:8019:8019 \
  -v "$MODEL_DIR:/models/chimera:ro" \
  -v "$TRANSFORMERS_DIR:/workspace/repos/transformers:ro" \
  -e PYTHONPATH=/workspace/repos/transformers/src \
  vllm/vllm-openai:gemma4 \
  --model /models/chimera \
  --served-model-name chimera \
  --model-impl transformers \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8019 \
  --data-parallel-size 4 \
  --api-server-count 1 \
  --aggregate-engine-logging \
  --max-num-batched-tokens 32768 \
  --performance-mode throughput \


echo "Container started. You can follow logs with: docker logs -f chimera-vllm"
