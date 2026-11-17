"""Probe sets.

The functional half of a scan depends on what the model is doing, so the prompt
set *is* an experimental condition -- the equivalent of the task a subject
performs inside the scanner. Scanning the same model under two sets and
subtracting gives a contrast map (see ``nnviz.contrast``), which is how task
fMRI localises function.
"""

from __future__ import annotations

PROMPT_SETS: dict[str, list[str]] = {
    "general": [
        "The capital of France is Paris, and the capital of Japan is",
        "She opened the door slowly, afraid of what might be waiting on the other side.",
        "In 1969, Apollo 11 landed on the Moon, carrying three astronauts:",
        "A balanced diet includes vegetables, whole grains, and sources of protein such as",
        "Dear Dr. Mensah, thank you for your note regarding the quarterly figures.",
        "The difference between weather and climate is that weather describes",
    ],
    "code": [
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2",
        "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2024-01-01' GROUP BY",
        "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):\n    def __init__(self, dim, heads):",
        "// Fix the off-by-one error in this loop\nfor (int i = 0; i <= n; i++) {\n    sum += values[i];\n}",
        "$ git rebase -i HEAD~3\nerror: cannot rebase: You have unstaged changes.",
    ],
    "math": [
        "Compute 17 * 24. First, 17 * 20 = 340, and 17 * 4 = 68, so the answer is",
        "If 3x + 7 = 28, then 3x = 21, so x =",
        "The derivative of x^3 * sin(x) with respect to x is, by the product rule,",
        "A fair coin is flipped 4 times. The probability of getting exactly two heads is",
        "7, 14, 28, 56, 112, and the next number in the sequence is",
    ],
    "multilingual": [
        "La bibliothèque ouvre ses portes à neuf heures du matin.",
        "Die Wissenschaftlerin erklärte, dass die Ergebnisse überraschend waren.",
        "今天天气很好，我们一起去公园散步吧。",
        "El año pasado visitamos las ruinas cerca de la costa.",
        "البحث العلمي يحتاج إلى صبر ودقة في القياس.",
        "ঢাকার রাস্তায় বর্ষাকালে প্রচুর জল জমে যায়।",
    ],
    "story": [
        "Once upon a time, in a village at the edge of a frozen lake, there lived a clockmaker who",
        "The letter arrived twenty years late, and by then the house had changed hands three times.",
        "He had never seen the sea, so when the train finally cleared the hills he stood up without thinking,",
    ],
    "reasoning": [
        "All roses are flowers. Some flowers fade quickly. Therefore, can we conclude that some roses fade quickly? Let us reason step by step.",
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully:",
        "If it is raining then the ground is wet. The ground is not wet. Therefore",
    ],
}

DEFAULT_SET = "general"

# The sentence played in the dynamic scan unless you pass your own (--text).
# Short, concrete, with a mid-sentence turn ("but") and a callback ("it"), so
# there is something to watch the model do token by token.
DEFAULT_TRACE = ("The cat saw a bird outside the window, jumped off the sofa and ran "
                 "to the door, but the door was closed, so it sat down and waited.")


def resolve(spec: str | list[str] | None) -> tuple[list[str], str]:
    """Resolve a prompt-set name, an explicit list, or ``None`` -> (prompts, label).

    ``"all"`` concatenates every set, which gives the most task-neutral anatomy.
    Comma-separated names are merged, e.g. ``"code,math"``.
    """
    if spec is None:
        return list(PROMPT_SETS[DEFAULT_SET]), DEFAULT_SET
    if isinstance(spec, (list, tuple)):
        return list(spec), "custom"
    name = str(spec).strip()
    if name == "all":
        merged: list[str] = []
        for key in PROMPT_SETS:
            merged.extend(PROMPT_SETS[key])
        return merged, "all"
    parts = [p.strip() for p in name.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PROMPT_SETS]
    if unknown:
        raise KeyError(f"unknown prompt set(s) {unknown}; available: {sorted(PROMPT_SETS)}")
    merged = []
    for p in parts:
        merged.extend(PROMPT_SETS[p])
    return merged, ",".join(parts)
