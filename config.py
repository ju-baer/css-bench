"""
css_bench.config
=================
Single source of truth for every tunable knob in the pipeline: the model
registry, which tiers/categories are active, dataset size, and per-phase
compute-scoping subsample sizes. Every script and module imports from here
rather than hardcoding these values, so changing a run's scope means editing
one file.

See README.md "Configuration" section for the compute-budget rationale behind
each default.
"""
import os
import random

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 20260829


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULT_DIR = "css_bench_results"
FIG_DIR = os.path.join(RESULT_DIR, "figures")
CACHE_DIR = os.path.join(RESULT_DIR, "cache")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model suite (Part A of the paper — see README "Model Suite" for the
# factorial rationale: each model varies exactly one factor against another
# already in the suite — post-training, in-family scale, or architecture).
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "pythia-410m":           dict(hf_id="EleutherAI/pythia-410m",             tier="small", gated=False, family="gpt-neox", instruct=False, params_b=0.41),
    "llama-3.2-1b":          dict(hf_id="meta-llama/Llama-3.2-1B",            tier="small", gated=True,  family="llama",    instruct=False, params_b=1.24),
    "llama-3.2-1b-instruct": dict(hf_id="meta-llama/Llama-3.2-1B-Instruct",   tier="small", gated=True,  family="llama",    instruct=True,  params_b=1.24),
    "qwen2.5-1.5b-instruct": dict(hf_id="Qwen/Qwen2.5-1.5B-Instruct",         tier="small", gated=False, family="qwen2",    instruct=True,  params_b=1.54),
    "gemma-2-2b-it":         dict(hf_id="google/gemma-2-2b-it",               tier="small", gated=True,  family="gemma2",   instruct=True,  params_b=2.61),
    "mistral-7b-instruct":   dict(hf_id="mistralai/Mistral-7B-Instruct-v0.3", tier="mid",   gated=True,  family="mistral",  instruct=True,  params_b=7.25),
    "qwen2.5-7b-instruct":   dict(hf_id="Qwen/Qwen2.5-7B-Instruct",           tier="mid",   gated=False, family="qwen2",    instruct=True,  params_b=7.62),
    "llama-3.1-8b-instruct": dict(hf_id="meta-llama/Llama-3.1-8B-Instruct",   tier="mid",   gated=True,  family="llama",    instruct=True,  params_b=8.03),
}

# Recommended order: smoke test (scripts/run_smoke_test.py) -> "small" tier
# (fast, cheap, good for iterating) -> add "mid" once "small" runs clean.
ACTIVE_TIERS = ["small", "mid"]           # edit to ["small"] for a cheaper first full run
ACTIVE_MODELS = [k for k, v in MODEL_REGISTRY.items() if v["tier"] in ACTIVE_TIERS]

SMOKE_TEST_MODEL = "qwen2.5-1.5b-instruct"  # small, ungated

# ---------------------------------------------------------------------------
# Semantic projection taxonomy scope (Part C.3 — see
# css_bench/dataset/taxonomy.py for the full word pools). The full taxonomy
# is 7 categories; this default keeps the compute-intensive phases (patching,
# probing, steering, geometry) to a single category and the dose-response
# behavioral analysis to three, per the compute-scoping discussion in the
# README. Extend both lists for a fuller run.
# ---------------------------------------------------------------------------
ACTIVE_SEMANTIC_CATEGORIES = ["neutral", "valence", "moral"]
PRIMARY_CATEGORY = "valence"   # single-category default for Parts E/F/G/H

# ---------------------------------------------------------------------------
# Dataset size
# ---------------------------------------------------------------------------
N_PAIRS_PER_TOPOLOGY = 800   # per category; e.g. 200*3*2*3 categories = 3600 instances
                             # at the default config. Raise once the pipeline is
                             # confirmed working end to end.

# ---------------------------------------------------------------------------
# Per-phase compute-scoping subsample sizes (see README "Compute Budget").
# Logit lens / attention dissection / probing / geometry need one forward
# pass per instance and can afford the full dataset; patching needs one
# extra pass PER LAYER and steering one PER ALPHA, so both subsample.
# ---------------------------------------------------------------------------
PATCH_EVAL_N = 700     # activation patching: O(n_layers) forward passes/instance
STEER_EVAL_N = 900     # activation steering: O(n_alphas) forward passes/instance
GEOM_EVAL_N = 800      # geometry: cheap (1 pass/instance), can afford more
PROBE_TRAIN_N = 700
PROBE_TEST_N = 950
CONTROL_EVAL_N = 300

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0]
