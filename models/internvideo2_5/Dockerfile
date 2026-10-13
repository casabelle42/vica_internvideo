# InternVideo2.5 Chat 8B, NVIDIA Thor / ARM64
FROM nvcr.io/nvidia/pytorch:25.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/hf-cache \
    MODEL_PATH=/opt/models/InternVideo2_5_Chat_8B

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Preserve the NVIDIA-provided CUDA torch and torchvision.
# These are the packages validated in the working Thor container.
RUN python3 -m pip install --no-cache-dir \
    'transformers==4.40.1' accelerate timm einops sentencepiece \
    safetensors pillow imageio av opencv-python-headless

RUN mkdir -p /opt/models/InternVideo2_5_Chat_8B \
    /opt/models/internvideo_tokenizer_test /opt/scripts/internvideo2_5
COPY models/ /opt/models/InternVideo2_5_Chat_8B/

# Reproduce the tested tokenizer workaround, keeping the upstream file intact.
# WARNING: token 354 is changed from an actual NUL to a diagnostic substitute.
# Normal text and one video inference worked; complete tokenizer equivalence
# is NOT established. Retain the original tokenizer for future remediation.
RUN python3 - <<'PY'
from pathlib import Path
import shutil
from sentencepiece import sentencepiece_model_pb2 as pb

source = Path('/opt/models/InternVideo2_5_Chat_8B')
target = Path('/opt/models/internvideo_tokenizer_test')
for name in ('tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
             'tokenization_internlm2.py', 'configuration_internlm2.py'):
    if (source / name).exists():
        shutil.copy2(source / name, target / name)
model = pb.ModelProto()
model.ParseFromString((source / 'tokenizer.model').read_bytes())
assert len(model.pieces) == 92544, 'Unexpected tokenizer vocabulary size'
assert model.pieces[354].piece == '\x00', 'Upstream tokenizer differs; review patch'
model.pieces[354].piece = '<|nul_test|>'
(target / 'tokenizer.model').write_bytes(model.SerializeToString())
print('Patched tokenizer prepared; original preserved')
PY

COPY run_internvideo2_5.sh /usr/local/bin/run_internvideo2_5.sh

RUN chmod +x /usr/local/bin/run_internvideo2_5.sh

ENTRYPOINT ["/usr/local/bin/run_internvideo2_5.sh"]
