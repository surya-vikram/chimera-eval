#!/bin/bash

# Usage: ./serve_chimera.sh /path/to/model_dir
# If no argument is provided, it defaults to the test path.

MODEL_DIR=${1:-"/nvme_zone3/home/ekamai1/chimera/data/exports/zoro2_v2_full"}
TRANSFORMERS_DIR="/nvme_zone3/home/ekamai1/chimera/repos/transformers"
# Chimera ends each chat turn with <end_of_turn>, but its generation_config.json lists only <EOS>
# as eos_token_id, so vLLM would keep generating past the end of the turn. Serve a copy that lists
# both: vLLM stops at every generation-config eos_token_id and leaves it out of the returned text.
GENERATION_DIR=${GENERATION_DIR:-"$HOME/.cache/chimera-vllm/generation-$(basename "$MODEL_DIR")"}

echo "Serving vLLM with model from: $MODEL_DIR"
echo "Using transformers from: $TRANSFORMERS_DIR"

mkdir -p "$GENERATION_DIR"
python3 - "$MODEL_DIR" "$GENERATION_DIR/generation_config.json" <<'PY' || exit 1
import json, sys
model, target = sys.argv[1:]
config = json.load(open(f'{model}/generation_config.json'))
tokens = {t['content']: t['id'] for t in json.load(open(f'{model}/tokenizer.json'))['added_tokens']}
eos = config.get('eos_token_id')
eos = [eos] if isinstance(eos, int) else list(eos or [])
config['eos_token_id'] = eos + [tokens[t] for t in ('<EOS>', '<end_of_turn>') if tokens[t] not in eos]
json.dump(config, open(target, 'w'), indent=2)
print('Stop tokens (eos_token_id):', config['eos_token_id'])
PY

# Clean up existing container
docker rm -f chimera-vllm 2>/dev/null

docker run -d \
  --name chimera-vllm \
  --gpus '"device=4,5,6,7"' \
  --ipc=host \
  -p 127.0.0.1:8019:8019 \
  -v "$MODEL_DIR:/models/chimera:ro" \
  -v "$GENERATION_DIR:/models/chimera-generation:ro" \
  -v "$TRANSFORMERS_DIR:/workspace/repos/transformers:ro" \
  -e PYTHONPATH=/workspace/repos/transformers/src \
  vllm/vllm-openai:gemma4 \
  --model /models/chimera \
  --generation-config /models/chimera-generation \
  --served-model-name chimera \
  --model-impl transformers \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8019 \
  --api-server-count 1 \
  --aggregate-engine-logging \
  --max-num-batched-tokens 32768 \
  --performance-mode throughput

echo "Container started. You can follow logs with: docker logs -f chimera-vllm"
