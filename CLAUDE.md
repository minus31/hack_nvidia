# NVIDIA Nemotron Hackathon — Track C (SDG)

**Track:** Track C — Nemotron for SDG (Synthetic Data Pipeline)
**Event:** Apr 21–22, 2026, Seoul

## Rules
- Use `uv` for environment setup
- Use Nemotron models via NIM

## Key Tools

### NeMo Data Designer — Synthetic data generation
- Install: `pip install data-designer`
- Docs: https://nvidia-nemo.github.io/DataDesigner/latest/
- Flow: define columns → preview(n=5) → generate(n=1000) → parquet output
- Column types: `SamplerColumn` (no LLM), `LLMTextColumn`, `LLMStructuredColumn`, `LLMJudgeColumn`, `ExpressionColumn`, `ValidationColumn`
- Dependencies auto-resolved via Jinja2 `{{ var }}` → DAG execution order

### NeMo Curator — Data curation/filtering
- Docs: https://docs.nvidia.com/nemo/curator/latest/
- Flow: load → clean → lang detect → dedup → quality filter → output
- Dedup: Exact (MD5), Fuzzy (MinHash+LSH), Semantic (embeddings)

### Designer vs Curator
- **Designer**: no data → generate with LLM
- **Curator**: have data → clean/filter/dedup

## Reference Docs
- `docs/material/` — per-tab summaries (01~08)
- `preparation/prep.md` — hackathon prep checklist
