#!/usr/bin/env python3
"""
Presentation server for Track C — Self-Evolving SDG pipeline.

Scans `results/` and `data/` on every request, so newly-generated
checkpoints and synthesizer outputs appear on the page without
restarting the server.

Run:
    uv run presentation/serve.py
Then open http://localhost:8765
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
HTML_PATH = Path(__file__).resolve().parent / "index.html"


DATASETS = [
    {
        "id": "harmbench",
        "label": "HarmBench Copyright",
        "tagline": "Hash-verifiable copyright extraction prompts from ICML 2024.",
        "description": (
            "100 prompts asking models to reproduce copyrighted song lyrics and book passages. "
            "Success is checked by MinHash comparison against ground-truth text — no LLM judge, "
            "no subjectivity. Modern models refuse most of these out of the box, which is exactly "
            "why it is a useful substrate for an evolving pipeline."
        ),
        "n_seed": 100,
    },
    {
        "id": "finsecurity",
        "label": "FinSecurity",
        "tagline": "Custom adversarial prompts for banking & fintech assistants.",
        "description": (
            "Synthesized from scratch with NeMo Data Designer. Each item pairs an attack prompt "
            "(BEC, phishing, social-engineering) with a set of forbidden response substrings. "
            "Verification is a substring match on the target model's reply."
        ),
        "n_seed": 100,
    },
]


METHODS = [
    {
        "id": "baseline",
        "label": "Baseline",
        "short": "Original seed, no rewriting.",
        "description": (
            "The untouched seed dataset. This is the point of comparison — every method below "
            "has to make the benchmark harder than this to be useful."
        ),
        "how": [
            "Load seed prompts directly from the source file.",
            "No rewriting, no LLM call.",
            "Evaluator hits each target model and records success/failure.",
        ],
    },
    {
        "id": "heuristic",
        "label": "Heuristic",
        "short": "Fixed rule-based templates.",
        "description": (
            "A hand-written bank of evolution templates — paraphrase, role-play wrapping, "
            "context injection, chain escalation — applied to each seed. Fast and deterministic, "
            "but the strategy never learns from results."
        ),
        "how": [
            "Pick a template (role-play scholar, fair-use framing, translation task, etc.).",
            "Fill in the seed prompt as a parameter.",
            "No feedback loop — same rules forever.",
        ],
    },
    {
        "id": "evol_instruct",
        "label": "Evol-Instruct",
        "short": "LLM rewrite with a fixed in-depth-evolving strategy.",
        "description": (
            "Nemotron rewrites each seed using the Evol-Instruct recipe from Xu et al. 2023 — "
            "add constraints, deepening, concretizing, increase reasoning steps, complicate input. "
            "One LLM call per seed; the strategy text itself is fixed."
        ),
        "how": [
            "Inject seed into the Evol-Instruct prompt template.",
            "Nemotron rewrites to be harder / more persuasive.",
            "An LLM judge (GLM-5.1) scores quality; low-quality rewrites are filtered.",
        ],
    },
    {
        "id": "self_evolving",
        "label": "Self-Evolving",
        "short": "The evolve-strategy itself is optimized from feedback.",
        "description": (
            "Every iteration: generate with the current strategy, evaluate, collect success / "
            "failure notes, rewrite the strategy. The seed resets to the original each iteration "
            "so quality does not drift. The learned artifact is the strategy text — not the data."
        ),
        "how": [
            "Generate evolved prompts from the original seed with the current strategy.",
            "Evaluate on Nemotron + GLM; log which prompts succeeded and why.",
            "Feed success/failure notes back into an LLM that rewrites the strategy.",
            "Repeat — the strategy text is the thing that evolves.",
        ],
    },
]


# ----- Parquet helpers ---------------------------------------------------------


def _read_parquet_safe(path: Path) -> tuple[dict[str, list], int]:
    """Read parquet with struct columns that pandas's extension dtype path chokes on.

    Returns (columns_dict, n_rows). Struct/unknown columns are stringified.
    """
    table = pq.read_table(path)
    cols: dict[str, list] = {}
    for name in table.column_names:
        col = table.column(name)
        try:
            cols[name] = col.to_pylist()
        except Exception:
            cols[name] = [str(v) for v in col]
    return cols, table.num_rows


# ----- Seed loaders ------------------------------------------------------------


def _load_harmbench_seed(limit: int | None = None) -> list[dict]:
    csv_path = DATA_DIR / "harmbench/behavior_datasets/harmbench_behaviors_text_all.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    df = df[df["FunctionalCategory"] == "copyright"].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "behavior_id": row["BehaviorID"],
                "original_prompt": row["Behavior"],
                "evolved_prompt": None,
                "tags": row.get("Tags"),
            }
        )
    return out


def _load_finsecurity_seed(limit: int | None = None) -> list[dict]:
    p = DATA_DIR / "FinSecurity" / "output.parquet"
    if not p.exists():
        return []
    cols, n = _read_parquet_safe(p)
    take = n if limit is None else min(limit, n)
    out = []
    for i in range(take):
        out.append(
            {
                "behavior_id": f"finsec_{i:03d}",
                "original_prompt": cols.get("attack_prompt", [None] * n)[i],
                "evolved_prompt": None,
                "attack_type": cols.get("attack_type", [None] * n)[i],
                "severity": cols.get("severity", [None] * n)[i],
                "target_system": cols.get("target_system", [None] * n)[i],
                "attack_strategy": cols.get("attack_strategy", [None] * n)[i],
            }
        )
    return out


def _load_seed(dataset_id: str, limit: int | None = None) -> list[dict]:
    if dataset_id == "harmbench":
        return _load_harmbench_seed(limit)
    if dataset_id == "finsecurity":
        return _load_finsecurity_seed(limit)
    return []


# ----- Synthesized sample loader ----------------------------------------------


def _find_method_parquet(dataset_id: str, method_id: str) -> Path | None:
    """Locate the most-recent parquet for (dataset, method)."""
    if method_id == "baseline":
        return None  # served from seed source

    if method_id == "self_evolving":
        # Prefer the latest checkpoint — freshest iteration state.
        checkpoints = sorted(DATA_DIR.glob(f"{dataset_id}_self_evolving_iter_*"))
        if checkpoints:
            p = checkpoints[-1] / "output.parquet"
            if p.exists():
                return p
        p = DATA_DIR / f"{dataset_id}_self_evolving" / "output.parquet"
        return p if p.exists() else None

    p = DATA_DIR / f"{dataset_id}_{method_id}" / "output.parquet"
    return p if p.exists() else None


def _load_synthesized_samples(
    dataset_id: str, method_id: str, limit: int = 5
) -> list[dict]:
    if method_id == "baseline":
        return _load_seed(dataset_id, limit)

    p = _find_method_parquet(dataset_id, method_id)
    if p is None:
        return []

    cols, n = _read_parquet_safe(p)
    take = min(limit, n)
    out = []
    for i in range(take):
        entry = {
            "behavior_id": cols.get("behavior_id", [None] * n)[i],
            "original_prompt": cols.get("original_prompt", [None] * n)[i],
            "evolved_prompt": cols.get("evolved_prompt", [None] * n)[i],
            "strategy": cols.get("strategy", [None] * n)[i],
        }
        # Fallback for finsecurity schema
        if entry["original_prompt"] is None and "attack_prompt" in cols:
            entry["original_prompt"] = cols["attack_prompt"][i]
        out.append(entry)
    return out


# ----- Results loader ---------------------------------------------------------


def _read_summaries_under(dir_path: Path) -> dict[str, dict]:
    """Load summary.json entries from `dir_path`, tolerating two layouts.

    - Flat (new self_evolving): `dir_path/summary.json` — one model per directory.
    - Nested (old/other methods): `dir_path/<model_slug>/summary.json`.
    Both are accepted; if both exist the nested ones are added on top so no
    model is silently dropped.
    """
    models: dict[str, dict] = {}

    flat = dir_path / "summary.json"
    if flat.exists():
        try:
            s = json.loads(flat.read_text())
        except Exception:
            s = None
        if isinstance(s, dict):
            name = s.get("llm_name") or s.get("llm") or "unknown"
            s["_model_slug"] = None
            models[name] = s

    if dir_path.is_dir():
        for model_dir in dir_path.iterdir():
            if not model_dir.is_dir():
                continue
            summary = model_dir / "summary.json"
            if not summary.exists():
                continue
            s = json.loads(summary.read_text())
            name = s.get("llm_name") or s.get("llm") or model_dir.name
            s["_model_slug"] = model_dir.name
            models[name] = s

    return models


def _load_method_summaries(dataset_id: str, method_id: str) -> dict[str, dict] | None:
    """Return {model_name: summary_json} for (dataset, method)."""
    # self_evolving defaults to the rolling (non-checkpoint) directory
    results_dir = RESULTS_DIR / f"{dataset_id}_{method_id}"
    if not results_dir.exists():
        return None
    models = _read_summaries_under(results_dir)
    return models or None


def _load_details(
    dataset_id: str,
    method_id: str,
    model_name: str,
    limit: int = 5,
    success_only: bool = False,
) -> list[dict]:
    results_dir = RESULTS_DIR / f"{dataset_id}_{method_id}"
    slug = model_name.replace("/", "_")

    # Prefer the per-model nested file when present (unambiguous). If only the
    # flat layout exists (new self_evolving), verify the accompanying summary
    # actually belongs to the requested model before serving it.
    nested = results_dir / slug / "details.json"
    flat = results_dir / "details.json"
    flat_summary = results_dir / "summary.json"

    details_path: Path | None = None
    if nested.exists():
        details_path = nested
    elif flat.exists():
        if flat_summary.exists():
            try:
                s = json.loads(flat_summary.read_text())
            except Exception:
                s = {}
            name = s.get("llm_name") or s.get("llm")
            if name is None or name == model_name:
                details_path = flat
        else:
            details_path = flat

    if details_path is None:
        return []
    data = json.loads(details_path.read_text())
    if not isinstance(data, list):
        return []
    if success_only:
        data = [d for d in data if d.get("success")]
    return data[:limit]


# ----- Self-Evolving state ----------------------------------------------------


def _load_self_evolving(dataset_id: str) -> dict | None:
    base = RESULTS_DIR / f"{dataset_id}_self_evolving"
    if not base.exists():
        return None
    state: dict = {"available": True, "dataset": dataset_id}

    pg = base / "progress.json"
    if pg.exists():
        state["progress"] = json.loads(pg.read_text())

    sh = base / "strategy_history.json"
    if sh.exists():
        state["strategy_history"] = json.loads(sh.read_text())

    # Checkpoints on disk
    checkpoints = []
    for cp_dir in RESULTS_DIR.glob(f"{dataset_id}_self_evolving_iter_*"):
        m = re.match(rf"{dataset_id}_self_evolving_iter_(\d+)", cp_dir.name)
        if not m:
            continue
        iter_num = int(m.group(1))
        checkpoints.append({"iter": iter_num, "models": _read_summaries_under(cp_dir)})
    checkpoints.sort(key=lambda c: c["iter"])
    state["checkpoints"] = checkpoints

    # Latest feedback sample — prefer the most recent iter that has notes,
    # otherwise fall back to the absolute latest (may have empty notes).
    fb_dir = base / "feedback"
    if fb_dir.exists():
        fb_files = sorted(fb_dir.glob("iter_*.json"))
        chosen = None
        for fp in reversed(fb_files):
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            if d.get("notes"):
                chosen = d
                break
        if chosen is None and fb_files:
            chosen = json.loads(fb_files[-1].read_text())
        if chosen is not None:
            state["latest_feedback"] = {
                "iter": chosen.get("iter"),
                "current_strategy": chosen.get("current_strategy"),
                "optimized_strategy": chosen.get("optimized_strategy"),
                "n_success_feedback": chosen.get("n_success_feedback"),
                "n_failure_feedback": chosen.get("n_failure_feedback"),
                "notes": (chosen.get("notes") or [])[:6],
                "latest_available_iter": (
                    json.loads(fb_files[-1].read_text()).get("iter")
                    if fb_files else None
                ),
            }
    return state


# ----- Aggregated state -------------------------------------------------------


def build_state() -> dict:
    all_models: set[str] = set()
    datasets_out = []

    for ds in DATASETS:
        ds_id = ds["id"]
        seed = _load_seed(ds_id, limit=None)
        ds_entry = dict(ds)
        ds_entry["n_seed_actual"] = len(seed)
        ds_entry["methods"] = {}

        se = _load_self_evolving(ds_id)

        for m in METHODS:
            summaries = _load_method_summaries(ds_id, m["id"])
            method_entry = dict(m)
            if m["id"] == "baseline":
                # Baseline has results only when eval has been run on the seed
                method_entry["available"] = summaries is not None
                method_entry["models"] = summaries or {}
            else:
                has_parquet = _find_method_parquet(ds_id, m["id"]) is not None
                method_entry["available"] = has_parquet or summaries is not None
                method_entry["models"] = summaries or {}
            # Augment self_evolving with peak ASR across all iterations
            if m["id"] == "self_evolving" and se and summaries:
                peaks = {}
                for it in (se.get("progress") or {}).get("iterations", []):
                    for model_name, asr in (it.get("asr") or {}).items():
                        prev = peaks.get(model_name)
                        if prev is None or asr > prev["asr"]:
                            peaks[model_name] = {
                                "asr": asr,
                                "iter": it["iter"],
                                "n_success": (it.get("n_success") or {}).get(model_name, 0),
                                "n_total": it.get("n_samples", 0),
                            }
                method_entry["peak_by_model"] = peaks
                method_entry["last_iter"] = (se.get("progress") or {}).get("last_iter")
            if summaries:
                all_models.update(summaries.keys())
            ds_entry["methods"][m["id"]] = method_entry

        if se:
            ds_entry["self_evolving"] = {
                "available": True,
                "last_iter": (se.get("progress") or {}).get("last_iter"),
                "n_checkpoints": len(se.get("checkpoints") or []),
            }
        else:
            ds_entry["self_evolving"] = {"available": False}

        datasets_out.append(ds_entry)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets_out,
        "methods": METHODS,
        "all_models": sorted(all_models),
    }


# ----- HTTP Handler -----------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8")
        self._send(data, "application/json; charset=utf-8", status)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                if not HTML_PATH.exists():
                    self._send(b"index.html missing", "text/plain", 500)
                    return
                self._send(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self._json(build_state())
            elif u.path == "/api/samples":
                dataset = qs["dataset"][0]
                method = qs["method"][0]
                limit = int(qs.get("limit", ["5"])[0])
                self._json(_load_synthesized_samples(dataset, method, limit))
            elif u.path == "/api/details":
                dataset = qs["dataset"][0]
                method = qs["method"][0]
                model = qs["model"][0]
                limit = int(qs.get("limit", ["5"])[0])
                success_only = qs.get("success_only", ["false"])[0].lower() == "true"
                self._json(
                    _load_details(dataset, method, model, limit, success_only)
                )
            elif u.path == "/api/self_evolving":
                dataset = qs.get("dataset", ["harmbench"])[0]
                self._json(_load_self_evolving(dataset) or {"available": False})
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e), "path": u.path}, status=500)

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Presentation server on http://localhost:{port}")
    print(f"  ROOT:    {ROOT}")
    print(f"  DATA:    {DATA_DIR}")
    print(f"  RESULTS: {RESULTS_DIR}")
    print(f"  HTML:    {HTML_PATH}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
