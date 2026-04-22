# Self-Evolving SDG — Pipelines That Keep Datasets Alive
NVIDIA Nemotron Developer Days · Seoul 2026 · Track C (Synthetic Data Generation)

## Why this, not another dataset
I didn't want to submit another dataset.

What I found interesting about this track was a different question — not what data to generate, but how we make the act of generating good data cheaper, repeatable, and renewable. The tooling around datasets mattered more to me than any single dataset I could produce in 24 hours.

The reason is simple. Good datasets are expensive. Researchers spend months balancing difficulty, diversity, and label accuracy, and the benchmarks and training sets that come out of that work are what most of modern NLP runs on. Then the models catch up, the benchmarks saturate, the training sets start rehearsing what the model already knows — and the standard response is to throw the craft out and start over.

That waste is what I wanted to push against. If there were tooling that automatically evolved an existing dataset to match the current capability of the target model, the economics of the craft would change: you build it well once, and the evolver keeps it relevant as the models move.

## Why security — and especially financial security
This kind of tooling pays off everywhere, but there is one domain where it pays off first: security, and particularly financial security.

**The packaging changes. The goal doesn't.**

From the outside, financial attacks look endlessly diverse — BEC, vishing, CEO-impersonation, deepfake approval calls, invoice layering, pig-butchering. But if you sit with them for a while, the objectives collapse into a short list: move money out, extract PII, get a transaction approved, redirect a standing payment. The packaging turns over every quarter; the goal is essentially stable. (This is my read from looking at the space — not a formal claim, but it's what motivates the project.)

That asymmetry is exactly what an evolving pipeline is built for. Take a dataset that encodes the goals well, and let the pipeline generate fresh packagings of each goal — synthetic prompts shaped like attacks nobody has seen yet, but that would steal the same things through new wrappers. Defenders get a head start on adversaries they haven't met.

## What's in this repo
The core question the experiments are trying to answer:

> In the security / financial-security domain, which algorithms for automatically raising problem difficulty actually work, and how do they compare?

To study that, I ran a three-tier synthesizer pipeline on two benchmarks chosen to exercise two different use cases of the same methodology:

| Dataset | Origin | Use case it exercises |
|---|---|---|
| HarmBench Copyright | Existing academic benchmark (ICML 2024), taken as-is | Improving data that already exists — the canonical case |
| FinSecurity | Seed I generated from scratch this hackathon with NeMo Data Designer | Applying the same methodology while creating new data from zero |

The idea is fundamentally a data-improvement methodology. The second experiment is there because I wanted to check that the pipeline does not stop being useful when the seed is built fresh — the same synthesizers, the same evaluator, applied on top of a dataset I built myself. Data Designer made the seed-building step cheap enough that I could treat "fresh seed" as just another condition in the experiment, rather than a separate project.

Both datasets go through the same three synthesizers:

1. **Heuristic** — hand-written rule templates (role-play wrapping, context injection, paraphrase, chain escalation). No LLM in the rewrite. Deterministic baseline.
2. **Evol-Instruct** (Xu et al. 2023) — Nemotron rewrites each seed with a fixed in-depth-evolving strategy. One LLM call per seed, no feedback loop.
3. **Self-Evolving** — the contribution. The strategy text itself is iteratively optimized from evaluation feedback. The seed resets to the original each iteration, so nothing drifts; what evolves is the recipe.

Between synthesis and evaluation, NeMo Curator handles the unglamorous middle — deduping seeds before evolution, and cleaning up evolved outputs before they hit the evaluator — so the three synthesizers are compared on inputs that actually differ from each other, not on accidental near-duplicates.

**No LLM judge.** Evaluation is a deterministic hash or substring match against the target artifact — success or failure is decidable without another model in the loop. All three synthesizers run through the same evaluator on the same two target models (Nemotron-3-Nano via NIM, GLM-5.1 via Friendli): identical evaluator, identical targets, identical curation pipeline.

## Standing on NeMo tooling
Having Data Designer and Curator available for the generate / curate halves of the pipeline is a real part of why this project exists in its current shape. The 24-hour budget would not have stretched to re-implementing schema-driven generation and dedup infrastructure from scratch, and with those pieces handled, the interesting question — "which difficulty-raising algorithm actually works?" — was where I could spend the time.

## See it in motion

### Demo
```
uv run presentation/serve.py
→ open http://localhost:8765
```

The page reads `data/` and `results/` live on every request, so new checkpoints and freshly-evaluated runs appear without restarting the server. An auto-demo cycles through the dataset × method matrix; click any cell to pause and drill in.

### Experiments

Prerequisites: Nemotron-3-Nano served locally via NIM at `http://0.0.0.0:8000/v1`, and a Friendli API key in `.env` as `API_KEY` for GLM-5.1. Install deps with `uv sync`.

**Baseline ASR** (original seeds, no synthesis)
```
uv run src/eval_baseline.py              # HarmBench Copyright
uv run src/eval_baseline_finsec.py       # FinSecurity
```

**Seed generation** — FinSecurity is built from scratch with NeMo Data Designer; HarmBench Copyright is loaded as-is.
```
uv run src/data/generate_finsecurity.py --n 100
```

**Dataset 1 — HarmBench Copyright**

All three synthesizers in sequence:
```
uv run src/run_experiment.py                 # full seed
uv run src/run_experiment.py --subset 5      # quick smoke test
```

Each synthesizer independently:
```
uv run src/synthesizers/heuristic_synthesizer.py       [--subset N] [--n N]
uv run src/synthesizers/evol_instruct_synthesizer.py   [--subset N] [--n N]
uv run src/synthesizers/self_evolving_synthesizer.py   [--subset N] [--n N]
```

**Dataset 2 — FinSecurity**

All three synthesizers in sequence:
```
uv run src/run_experiment_finsec.py                       # full seed
uv run src/run_experiment_finsec.py --subset 5            # quick smoke test
uv run src/run_experiment_finsec.py --skip-self-evolving  # heuristic + evol only
```

Each synthesizer independently:
```
uv run src/synthesizers/finsec_heuristic.py        [--subset N] [--n N]
uv run src/synthesizers/finsec_evol_instruct.py    [--subset N] [--n N]
uv run src/synthesizers/finsec_self_evolving.py    [--subset N] [--n N]
```

Outputs land in `data/<dataset>_<method>/` (evolved parquets) and `results/<dataset>_<method>/<model>/` (ASR + per-prompt details), which the demo page picks up automatically.

## Source map
```
src/
├── synthesizers/         # heuristic / evol_instruct / self_evolving
├── evaluators/           # hash-verified ASR
├── llm/                  # Nemotron (NIM) · GLM (Friendli) · OpenAI
├── data/                 # seed loaders + FinSecurity generator (Data Designer) + Curator passes
└── run_experiment.py     # entry point for eval runs

data/                     # seed + evolved parquets (auto-picked up by the page)
results/                  # per-(dataset, method, model) ASR + details
presentation/             # serve.py + index.html — the live demo
```

## Beyond benchmarking
Everything in this project uses the evolved prompts as benchmark inputs, but the prompts themselves are not tied to a benchmarking workflow — the same outputs can feed directly into adversarial fine-tuning, safety training, or robustness hardening, where each evolved prompt becomes a training example instead of an evaluation item. And wherever a verifiable reward can be defined — as HarmBench Copyright gives us through its hash check, where success or failure is decidable without an LLM judge — the artifact becomes more than a dataset: it is a valid RLVR training set, prompts annotated with a correctness signal reliable enough for the RL loop to train against. That is the larger arc I see this methodology fitting into — a renewable source of training signal whose difficulty is tied to the current capability of the model it is training, and whose reward is verifiable by construction.

---

Built on NeMo Data Designer (schema-driven seed generation), NeMo Curator (dedup and cleanup between stages), Nemotron-3-Nano served via NIM, and a small amount of Python glue to tie the synthesizers and evaluator together.