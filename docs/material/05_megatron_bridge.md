# 05. Megatron-Bridge — Track B

> NeMo 프레임워크 내 PyTorch 네이티브 라이브러리. HuggingFace ↔ Megatron 체크포인트 변환 + SFT/LoRA 파인튜닝

## Track B 파이프라인에서의 위치
```
데이터 준비 (NeMo Data Designer 등)
    ↓
Megatron-Bridge (SFT / LoRA)   ← 여기
├── HF 체크포인트 → Megatron 변환
└── Supervised Fine-Tuning
    ↓
파인튜닝된 Nemotron Nano 체크포인트
    ↓
NeMo RL (GRPO 강화학습)
```

## 환경 설정
```bash
# Nemotron 3 Nano 전용 컨테이너 (권장)
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --gpus all --shm-size=64g \
  nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
```

## 핵심: AutoBridge API
```python
from megatron.bridge import AutoBridge

# HF → Megatron 변환
AutoBridge.import_ckpt(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "./megatron_checkpoints/nemotron_nano"
)

# Megatron 모델 프로바이더 생성
bridge = AutoBridge.from_hf_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
```

## SFT (Full Fine-Tuning)
```bash
# 공식 Nano SFT 레시피 실행 (최소 2 H100 노드, GPU 16개)
torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_nano.py \
  train.global_batch_size=128 \
  train.train_iters=100 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt
```

```python
# Python API로 직접 SFT
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_sft_config
from megatron.bridge.training import finetune

config = nemotron_3_nano_sft_config()
config.train.train_iters = 100
config.checkpoint.pretrained_checkpoint = "./megatron_checkpoints/nemotron_nano"
finetune(config)
```

## LoRA (경량 파인튜닝)
```bash
torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_nano.py \
  --peft lora \
  train.global_batch_size=128 \
  checkpoint.pretrained_checkpoint=/path/to/megatron/ckpt
```
- LoRA 기본 타겟 모듈: `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`
- 학습 후 베이스 모델에 병합: `python examples/peft/merge_lora.py ...`

## HF로 내보내기
```bash
python examples/conversion/convert_checkpoints.py export \
  --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --megatron-path ./checkpoints/sft/iter_0001000 \
  --hf-path ./exports/nemotron_nano_sft_hf
```

## AutoBridge API 요약
| 메서드 | 설명 |
|--------|------|
| `AutoBridge.import_ckpt(hf, meg)` | HF → MCore 변환 저장 |
| `AutoBridge.from_hf_pretrained(path)` | Bridge 인스턴스 생성 |
| `bridge.to_megatron_provider()` | Megatron 모델 프로바이더 생성 |
| `bridge.export_ckpt(meg, hf)` | MCore → HF 변환 저장 |

## 참고 링크
- 공식 문서: https://docs.nvidia.com/nemo/megatron-bridge/latest/
- GitHub: https://github.com/NVIDIA-NeMo/Megatron-Bridge
