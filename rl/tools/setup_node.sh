#!/bin/bash
# Prepare a GPU node for the study. Safe to rerun. Copy the repo to $ROOT/graft and the SFT adapter to
# $ROOT/models/countdown_sft_qwen3-8b_lora first, then:
#   ROOT=/scratch bash $ROOT/graft/rl/tools/setup_node.sh
# Then `source $ROOT/env.sh`. Use the same ROOT as the run being resumed: run settings hold absolute paths.
set -eu
ROOT=${ROOT:-/scratch}
export UV_CACHE_DIR=$ROOT/uvcache HF_HOME=$ROOT/hf CUDA_VISIBLE_DEVICES=

CUDA=$(ls -d /usr/local/cuda-13.* 2>/dev/null | tail -1 || true)
if [ -z "$CUDA" ]; then
  echo "B300 kernels need CUDA 13: sudo apt-get install -y --no-install-recommends cuda-toolkit-13-0" >&2
  exit 1
fi

cat > "$ROOT/env.sh" <<ENV
ulimit -n 65536
export RH_REPO_PATH=$ROOT/rl-rewardhacking GRAFT_COUNTDOWN_SFT_MODEL=$ROOT/models/countdown_sft_qwen3-8b
export HF_HOME=$ROOT/hf UV_CACHE_DIR=$ROOT/uvcache STUDY_ROOT=$ROOT/study
export CUDA_HOME=$CUDA PATH=\$HOME/.local/bin:$CUDA/bin:\$PATH
cd $ROOT/graft/rl
[ -f .venv/bin/activate ] && source .venv/bin/activate
ENV

which uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
[ -d "$ROOT/rl-rewardhacking" ] || git clone -q https://github.com/ariahw/rl-rewardhacking "$ROOT/rl-rewardhacking"
cd "$ROOT/graft/rl"
uv sync 2>&1 | tail -2
source .venv/bin/activate
bash vllm_patches/apply.sh 2>&1 | tail -1
[ -f "$ROOT/models/countdown_sft_qwen3-8b/model.safetensors" ] || python tools/merge_sft.py \
  --adapter "$ROOT/models/countdown_sft_qwen3-8b_lora" --out "$ROOT/models/countdown_sft_qwen3-8b"
echo NODE_READY
