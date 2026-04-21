# 06. NeMo RL — Track B

> SFT로 instruction-tuned된 Nemotron 3 Nano를 GRPO 강화학습으로 추가 정렬하는 Stage 2 가이드

## Track B 파이프라인에서의 위치
```
Megatron-Bridge (SFT)
    ↓
NeMo RL (GRPO 강화학습)   ← 여기
    ↓
최종 정렬된 모델
```

## 환경 설정
```bash
docker pull nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano
docker run --gpus all -it --rm -v /path/to/workspace:/workspace \
  nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano
```

## 학습 백엔드
| 백엔드 | 설명 | 적합한 경우 |
|--------|------|-------------|
| DTensor (FSDP2) | PyTorch 차세대 분산 학습, 설정 간단 | 소·중형 모델, 빠른 실험 |
| Megatron | TP/PP/EP 지원 고성능 학습 | 100B+ 대형 모델, 프로덕션 |

## 데이터 형식
```jsonl
{"input": "문제 텍스트", "output": "정답"}
```
- `DatumSpec` 인터페이스로 `message_log`, `extra_env_info`(정답) 포함

## GRPO 학습 실행
```bash
uv run python examples/run_grpo.py \
  --config my_grpo.yaml \
  policy.model_name=/path/to/hf_checkpoint \
  grpo.max_num_steps=1000 \
  grpo.num_prompts_per_step=64 \
  grpo.num_generations_per_prompt=8
```

## YAML 핵심 설정
```yaml
grpo:
  num_prompts_per_step: 128
  num_generations_per_prompt: 16
  max_num_steps: 1000000
  normalize_rewards: true

policy:
  model_name: "/path/to/hf_checkpoint"
  precision: "bfloat16"
  generation:
    backend: "vllm"
    vllm_cfg:
      gpu_memory_utilization: 0.5   # colocated 모드 시 0.5 이하 유지

loss_fn:
  reference_policy_kl_penalty: 0
  ratio_clip_min: 0.2
  ratio_clip_max: 0.28

cluster:
  gpus_per_node: 8
  num_nodes: 32
```

## 커스텀 보상 함수 구조
```python
@ray.remote(max_restarts=-1, max_task_retries=-1)
class MyRewardEnvironment(EnvironmentInterface):
    def step(self, message_log_batch, metadata):
        # 응답 추출, 보상 계산
        rewards = torch.tensor([[correctness, format_score], ...])  # shape [B, N]
        return EnvironmentReturn(rewards=rewards, terminateds=done, ...)
```
- 다중 보상: `rewards` shape `[B, N]` → N개 컴포넌트 독립 추적 가능

## 고급 기능

### Async GRPO
```yaml
grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 1   # 이전 스텝 trajectory 재사용으로 GPU 유휴 감소
```

### KL Penalty 조정
```bash
# 순수 GRPO (KL 없음)
loss_fn.reference_policy_kl_penalty=0

# KL 적용 (안정화)
loss_fn.reference_policy_kl_penalty=0.05
loss_fn.reference_policy_kl_type=k3   # k1, k2, k3 중 선택
```

## 모니터링 메트릭
| 메트릭 | 설명 | 권장 범위 |
|--------|------|-----------|
| token_mult_prob_error | 생성/학습 토큰 확률 오차 | < 1~2% |
| gen_kl_error | 생성 분포 KL 발산 | < 1e-3 |
| approx_entropy | 정책 엔트로피 (붕괴 감지) | 안정 유지 |
| sampling_importance_ratio | 분포 이동 보정 비율 | ≈ 1.0 |

## 모니터링 (TensorBoard)
```bash
# YAML에서 활성화
logger:
  tensorboard_enabled: true
  log_dir: "logs"

# 실행
tensorboard --logdir=./logs --port=6006
```

## 참고 링크
- 공식 문서: https://docs.nvidia.com/nemo/rl/latest/
- GitHub: https://github.com/NVIDIA-NeMo/RL
- GRPO 가이드: https://docs.nvidia.com/nemo/rl/latest/about/algorithms/grpo.html
