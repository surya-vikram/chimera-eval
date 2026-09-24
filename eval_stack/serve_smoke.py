"""Optional local GPU smoke server. Production evaluation assumes endpoints already exist."""
import argparse
from pathlib import Path
import subprocess

IMAGE = 'vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--port', type=int, default=8010)
    p.add_argument('--name', default='chimera-eval-smoke-vllm')
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    directory = Path(args.model_dir).resolve()
    snapshot_download('Qwen/Qwen3-0.6B', revision=REVISION, local_dir=directory,
                      allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','LICENSE','*.jinja'])
    subprocess.run(['docker','run','-d','--name',args.name,'--gpus','all','--ipc=host',
                    '-p',f'127.0.0.1:{args.port}:8000','-v',f'{directory}:/model:ro',IMAGE,
                    '/model','--served-model-name','eval-smoke','--max-model-len','8192',
                    '--max-num-seqs','4','--gpu-memory-utilization','0.80','--enforce-eager',
                    '--generation-config','vllm'],check=True)


if __name__ == '__main__': main()
