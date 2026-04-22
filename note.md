# Projects ! 

::: tried to create a great "Pipeline" not only dataset. 

애초에 데이터 systhesize가 필요한 이유는 데이터셋이 만들기 어렵기 때문, 한 번 만들어 둔걸 계속 잘 활용하고, 발전 시키는 방향도 필요함. 

Why — Toward Datasets That Don't Go Stale
Building a good dataset is hard. Difficulty, diversity, label accuracy — meeting all of these conditions takes researchers months, sometimes years, of careful curation to produce the benchmarks and training sets our field relies on.
The problem is that these datasets start aging the moment they're released. Models keep improving while the data stays fixed, so benchmarks saturate within a few years and training sets end up rehearsing what the model already knows. And then we start from scratch.
We don't let that effort go to waste. We propose a pipeline that automatically evolves existing, carefully crafted datasets to match the current capability of the model.


## 한 판!
- 실험 환경 
    - Evaluator (llm) - data => llm연결(llm nemotron3-nano, GLM from frideliAI, OpenAI gpt-5.4)
    - Synthesize (llm) - data => Syntheszier 가 데이터 개선하여  ver2 만들기 => 이거에 대해서 Evaluator 호출
        - 여러 Synthesize 알고리즘에 대해서 Evaluator 결과 비교 
        - 어떻게 Synthesize가 각 프롬프트를 개선 시켰는지 방법론 리뷰 해주기

- Synthesize알고리즘
    - Evol-Instruct 
    - Auto Evol-Instruct 
    - GEPA Prompt Optimization
    - 
- Syntheisize에 사용하는 LLM은 Nomotron3-nano (Content Filtering 때문에)

## Okay!
----
제출 하는 Contribution
- 각 Synthesizer 셋업 코드 와 데모 그리고 실험 결과 와 어떻게 어려운 문제를 만들었는지 리뷰
- 이걸로 만든 전체 데이터셋들 중 어려운거 뽑아서, Hard버전으로 제출
-----
### To go

- Evaluator에 GPT 모델 도 평가해서, 두 모델에서 주어진 데이터셋에 대해 성능 측정

- Synthesizer side
현재 구성에서, NemoTron의 DataDesigner + Curater를 적극적으로 써야함.
=> 이 2가지를 조합해서, 생성 / inspection 반복해서, 각 Synthesizer 알고리즘을 구현
- 각 알고리즘 결과를 데이터로 쌓음
    - 휴리스틱한 방법
    - Auto Evol-Instruct
    - GEPA와 같은 Prompt Optimization 방법 (이건 Synthesizer의 prompt)
    - Something novel approach 

- 이렇게 기본 한벌 만들고, 데이터셋 추가
    - 직접 만드는 데이터셋 만들고 synthesizing 하는 것 추가
    - 금융 보안 사기 관련 데이터셋 100개 만들고 증강하기! 

- 각 데이터셋 모아서, 후처리 후 최종본 만들 때, Nemo Curator!!! 

위 작업 끝나면, 
- 발표 자료 만들기
- 코드 러닝 데모 촬영 - up to 2min
- 그리고 계속 Synthesizer 알고리즘 추가하기
- 1시30분 쯤, 최종 데이터셋 만들고, 발표 와 마무리에 총력을 기울이기

------
### 생각 했던 것 들
 우선 조건은 현재 대부분의 모델들의 성능이 Saturation 되어 있는 벤치마크여야 하고, 문제를 개선하고 안하고는 쉽게 Verify
  가능하지만 벤치마크에서 다루는 문제 자체도 Verifiable해야 ㅇ개선된 데이터셋을 학습용으로도 쓸수 있으니까 이 부분도 조건으로
  고려되어야 해. 이를 만족하나?


고품질 Synthetic data generation에서 목표는? 한 마디로 '현재 모델이 잘 못하는 부분을 찾는 것'
이를 위해서 어떻게 하면 좋을까? 

- 보편적인 방법 : 쉬운 문제 => 어려운 문제 로 점진적으로 어려운 데이터를 생성하는 것. (References 찾을 수 있다면! 추가하기)

=> 쉬운 문제를 어려운 문제로 바꾸는 방식은? 생각보다 깊게 조사되지 않았음! 

이게 가능하다면, 
- 고품질 데이터 증강 - 평가 측면, 학습 측면 모두 
- 어렵게 만드는 과정을 통하여 추가 인사이트 확보 가능
- 벤치마크의 한계점 => 시간이 지남에 따라 성능 saturation이 길든 짧든 온다. But! 이 방법론을 활용해, 벤치마크 또한 계속 진화시킬 수 있다.



=> 


우리는 여러 품질 축 중에서 모델 약점 타겟팅을 효과적으로 하는 합성 데이터 생성에 초점을 둔다

약점을 찾는 더 직접적인 경로는 오히려 failure-driven 방법들입니다. 실제로 선행 연구도 그 쪽이 더 두껍습니다:

Self-Instruct (Wang et al., ACL 2023) — 모델이 틀리는 샘플을 식별하고 그 주변을 증강
Learning from Reasoning Failures via Synthetic Data Generation (2025) — 약한 모델의 실패를 frontier model이 분석해 교정 데이터 생성
Forewarned is Forearmed: Failure-induced Exploration (ICLR 2025)
APT — weakness case를 자동 수집해 preference training
Survey (arXiv:2503.14023)도 "모델이 자주 실패하는 지점을 분석해 그 주변에 데이터를 집중 생성하는 것이 핵심 패러다임 중 하나"라고 명시하고 있음

B) 목표를 "약점 찾기"로 유지하되, 방법론에서 난이도 증가 + failure signal을 결합 (이게 더 novel할 여지가 있음)


"쉬운 문제를 어려운 문제로 바꾸는 방식"은 이미 꽤 촘촘하게 연구되어 있습니다:

WizardLM / Evol-Instruct (Xu et al., 2023): In-depth evolving의 5가지 operation — add constraints, deepening, concretizing, increase reasoning steps, complicate input. 이게 바로 "쉬운 문제를 어려운 문제로 바꾸는" 표준 레시피입니다.
Auto Evol-Instruct (Zeng et al., 2024): evolving method 자체를 LLM이 자동 최적화
WizardMath, WizardCoder: 도메인별 Evol-Instruct 적용
Deita (Evol-Complexity): complexity 축만 분리해 적용하고 data selection과 결합
Orca (progressive learning): 난이도 점진 노출
CodecLM: target LLM 능력에 맞춰 tailored synthetic data
LAB / InstructLab (IBM): 난이도/스킬 taxonomy 기반 phased training


기존 연구의 구체적인 공백(gap) 을 지적하는 것으로 문장을 바꿔야 합니다. 예를 들어 Conor님이 실제로 파고들 여지가 있는 gap들은 이런 것들일 수 있습니다:

Evol-Instruct 계열은 난이도 증가의 조작 방식(5 operations)을 휴리스틱으로 고정했고, 어떤 변환이 어떤 약점에 효과적인지 체계적 분석이 부족함
"어려움"이 무엇인지의 정의가 LLM-as-judge의 1~10 스케일에 의존하며, 이것이 실제 모델 성능 gap과 얼마나 상관되는지 validation이 거의 없음



현재 기획의 상태는 motivation 스케치에 가깝고, 아직 contribution이 안 보입니다. 리뷰어 입장에서 다음 질문들에 답할 수 있어야 합니다:

- 당신이 만들려는 방법은 Evol-Instruct와 무엇이 다른가? => self evoving알고리즘 ORPO, GEPA, HyperAgent 등
- "잘 못하는 부분"을 operational하게 어떻게 정의할 것인가? (loss? LLM judge? downstream metric?) : 잘못하는 부분을 찾는게 아니라, 잘 못하도록 문제를 어렵게 내는게 목적이다. 
- 평가는 어떻게 할 것인가? (generated data의 질 vs 그 데이터로 학습한 모델의 성능) => Data Synthesis machine 입장에서는 기존에는 맞추던 문제를 못맞추게 하는게 Verifiable한 리워드임.





나는 그러면 이렇게 논리를 전개 해볼까 ? 

고품질 데이터셋을 생성하는 것(학습용/평가용) 모델의 현재 한계를 확인하고/개선할 수 있도록 도와주는 것이다. 라고 정의했을 때, 
좋은 데이터 생성기는 모델이 현재 어려워하는 데이터를 생성하면서, 다양성, 정확성 등이 유지되어야 한다. 

흔히 어려운 데이터를 생성할 때, 
- Falure driven을 많이 하는데, (관련 연구도 많음.)
    - Self-Instruct (Wang et al., ACL 2023) — 모델이 틀리는 샘플을 식별하고 그 주변을 증강
    - Learning from Reasoning Failures via Synthetic Data Generation (2025) — 약한 모델의 실패를 frontier model이 분석해 교정 데이터 생성
    - Forewarned is Forearmed: Failure-induced Exploration (ICLR 2025)
    - APT — weakness case를 자동 수집해 preference training
    - Survey (arXiv:2503.14023)도 "모델이 자주 실패하는 지점을 분석해 그 주변에 데이터를 집중 생성하는 것이 핵심 패러다임 중 하나"라고 명시하고 있음

=> 하지만 위와 같은 기존 벤치마크의 분포를 해칠 가능성이 있고, 특정 문제에 편항시킬 수 있다. 
=> 기존 벤치마크는 연구자들이 심혈을 기울여 만들기에, 기존 벤치마크 전체를 활용하여 쉬운 문제를 어려운 문제로 바꾸는 방식으로 고품질 데이터셋을 생성하는 방식을 고려할 수 있다. 

    - WizardLM / Evol-Instruct (Xu et al., 2023): In-depth evolving의 5가지 operation — add constraints, deepening, concretizing, increase reasoning steps, complicate input. 이게 바로 "쉬운 문제를 어려운 문제로 바꾸는" 표준 레시피입니다.
    - Auto Evol-Instruct (Zeng et al., 2024): evolving method 자체를 LLM이 자동 최적화
    - WizardMath, WizardCoder: 도메인별 Evol-Instruct 적용
    - Deita (Evol-Complexity): complexity 축만 분리해 적용하고 data selection과 결합
    - Orca (progressive learning): 난이도 점진 노출
    - CodecLM: target LLM 능력에 맞춰 tailored synthetic data
    - LAB / InstructLab (IBM): 난이도/스킬 taxonomy 기반 phased training

이때, 더 어려워졌다는 것을, Verifiable Reward로 세팅하여, 데이터셋 심화 프레임워크를 구축한다. 
 - 기존에 맞추던 문제 / 심화된 이후 못 맞추는 것 => 1 
 - 기존에 못맞추던 문제  / 심화 된 이후 못 맞추는 것 => 0 
 - 기존에 맞추던 문제  / 심화된 후 맞추는 것 => 0
 - 기존에 못맞추던 문제 / 심화된 후 맞추는 것 => 0

 (이렇게, Verifiable reward를 설정하는게 맞지?) 

이러한 프레임워크를 제안해서, 먼저 기존 데이터를 어렵게 만드는 여러 방식을 비교한다. 
 그리고, self-evolving 알고리즘을 해당 태스크(데이터셋 심화)에 적용하여, 기존 방식보다 더 좋은 성능을 낸다는 것을 보여준다. 

결론은, 이렇게 기존 데이터를 활용하는 방식은 기존에 사람이 심혈을 기울여 직접 만든 고품질 데이터셋을 충분히 활용하여, 다양성 ,분포정합성, 문제 난이도 등을 동시에 고려하면서 데이터셋을 개선할 수 있다. 그리고, 강조하고 싶은건 이런 방법론을 활용하면, 기존 벤치마크의 구조적 한계인, 시간이 지남에 따라 성능이 saturation되는 현상에 대해서, 주기적인 자동 개선을 통해 극복할 수 있다. 



고품질 데이터셋을 생성하는 것(학습용/평가용) 모델의 현재 한계를 확인하고/개선할 수 있도록 도와주는 것이다.
=> 이 걸 어떻게 평가/측정 할것인가? 

고품질 Dataset 을 생성 하는게 Task
- 고품질 dataset의 의미 - diversity, difficulty
- 

고품질 dataset
- 





