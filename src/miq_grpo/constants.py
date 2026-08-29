"""Pinned project constants and model-visible prompt policy."""

from __future__ import annotations

import hashlib

PROJECT_SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_POOL_ID = "ml-jku/moleculariq-trainPool"
OFFICIAL_BENCHMARK_ID = "ml-jku/moleculariq-v0.0"
OFFICIAL_TASK = "moleculariq_pass_at_k"

MOLECULARIQ_CORE_COMMIT = "a1b89635371c3cd942e44ebeec63ec3665e7743d"
MOLECULARIQ_EVAL_COMMIT = "425ecaaa8faf65aa43aa60ec0f584b7b7f060063"
OFFICIAL_TASK_YAML_SHA256 = "49c481683afec0dcca7de490ac778983e9ebddd750e5ec809377a07e6e67007d"
TRL_VERSION = "1.12.0"

TASK_FAMILIES = ("count", "index", "constraint")
BENCHMARK_DENYLIST = (
    "moleculariq-v0.0",
    "val_easy",
    "val_hard",
    "validation",
    "benchmark",
    "test_pool",
)

# Canonical public system instruction shipped by the pinned official evaluator.
# It is fixed before training and used unchanged for baseline and trained-model
# evaluation. Reward-only targets and constraints are never interpolated here.
SYSTEM_PROMPT = """You are an expert chemist. Answer molecular property, understanding, structural analysis and molecular generation questions precisely and accurately.

CRITICAL: Only content within <answer></answer> tags will be extracted. ALWAYS return JSON format.

KEY REQUIREMENT: Use EXACT key names from the question. Never modify or invent keys.

INDEXING: Atoms are indexed from 0 to the end of the SMILES string from left to right. Only heavy atoms (skip [H], include [2H]/[3H]).
Examples:
    - "CCO": C(0), C(1), O(2)
    - "CC(C)O": C(0), C(1), C(2), O(3)
    - "CC(=O)N": C(0), C(1), O(2), N(3)

ABSENT FEATURES: Use 0 for counts, [] for indices. Never null or omit.

ALWAYS USE JSON with EXACT keys from the question:

Single count (key from question: "alcohol_count"):
<answer>{"alcohol_count": 2}</answer>
<answer>{"alcohol_count": 0}</answer>  (if absent)

Single index (key from question: "ketone_indices"):
<answer>{"ketone_indices": [5]}</answer>
<answer>{"ketone_indices": []}</answer>  (if absent)

Multiple properties (keys from question: "ring_count", "halogen_indices"):
<answer>{"ring_count": 2, "halogen_indices": [3, 7]}</answer>
<answer>{"ring_count": 0, "halogen_indices": []}</answer>  (if all absent)

Constraint generation:
<answer>{"smiles": "CC(O)C"}</answer>

Include ALL requested properties. Never null or omit."""

SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
