"""
Fuzzy Logic Rules — Configurable rule base for the Mamdani inference system.

Rules are defined as data structures, not hard-coded in the UI.
Each rule maps input membership terms to an output risk level.
"""

from typing import List, Dict, Tuple


# ─────────────────────────────────────────────────────────
# Rule format:
#   Each rule is a dict with:
#     'antecedents': list of (variable_name, term_name) tuples
#     'consequent': (variable_name, term_name)
#     'weight': float (0–1), default 1.0
# ─────────────────────────────────────────────────────────

DEFAULT_RULES = [
    # Critical conditions
    {
        "antecedents": [("fault_probability", "high"), ("rms", "high")],
        "consequent": ("risk", "critical"),
        "weight": 1.0,
        "description": "High fault probability + high vibration → critical risk",
    },
    {
        "antecedents": [("rms", "high"), ("kurtosis", "high")],
        "consequent": ("risk", "critical"),
        "weight": 1.0,
        "description": "High vibration + high kurtosis → critical risk",
    },
    {
        "antecedents": [("fault_probability", "high"), ("kurtosis", "high")],
        "consequent": ("risk", "critical"),
        "weight": 0.9,
        "description": "High fault probability + high kurtosis → critical risk",
    },

    # High conditions
    {
        "antecedents": [("fault_probability", "high"), ("rms", "medium")],
        "consequent": ("risk", "high"),
        "weight": 1.0,
        "description": "High fault probability + medium vibration → high risk",
    },
    {
        "antecedents": [("fault_probability", "medium"), ("rms", "high")],
        "consequent": ("risk", "high"),
        "weight": 1.0,
        "description": "Medium fault probability + high vibration → high risk",
    },
    {
        "antecedents": [("fault_probability", "medium"), ("kurtosis", "high")],
        "consequent": ("risk", "high"),
        "weight": 0.9,
        "description": "Medium fault probability + high kurtosis → high risk",
    },
    {
        "antecedents": [("rms", "high"), ("kurtosis", "medium")],
        "consequent": ("risk", "high"),
        "weight": 0.8,
        "description": "High vibration + medium kurtosis → high risk",
    },

    # Medium conditions
    {
        "antecedents": [("fault_probability", "medium"), ("rms", "medium")],
        "consequent": ("risk", "medium"),
        "weight": 1.0,
        "description": "Medium fault probability + medium vibration → medium risk",
    },
    {
        "antecedents": [("fault_probability", "high"), ("rms", "low")],
        "consequent": ("risk", "medium"),
        "weight": 0.8,
        "description": "High fault probability + low vibration → medium risk",
    },
    {
        "antecedents": [("fault_probability", "low"), ("rms", "high")],
        "consequent": ("risk", "medium"),
        "weight": 0.8,
        "description": "Low fault probability + high vibration → medium risk",
    },
    {
        "antecedents": [("fault_probability", "medium"), ("rms", "low")],
        "consequent": ("risk", "medium"),
        "weight": 0.7,
        "description": "Medium fault probability + low vibration → medium risk",
    },
    {
        "antecedents": [("rms", "medium"), ("kurtosis", "medium")],
        "consequent": ("risk", "medium"),
        "weight": 0.7,
        "description": "Medium vibration + medium kurtosis → medium risk",
    },

    # Low conditions
    {
        "antecedents": [("fault_probability", "low"), ("rms", "low")],
        "consequent": ("risk", "low"),
        "weight": 1.0,
        "description": "Low fault probability + low vibration → low risk",
    },
    {
        "antecedents": [("fault_probability", "low"), ("rms", "medium")],
        "consequent": ("risk", "low"),
        "weight": 0.8,
        "description": "Low fault probability + medium vibration → low risk",
    },
    {
        "antecedents": [("fault_probability", "low"), ("kurtosis", "low")],
        "consequent": ("risk", "low"),
        "weight": 0.9,
        "description": "Low fault probability + low kurtosis → low risk",
    },
    {
        "antecedents": [("rms", "low"), ("kurtosis", "low")],
        "consequent": ("risk", "low"),
        "weight": 0.8,
        "description": "Low vibration + low kurtosis → low risk",
    },
]


def get_default_rules() -> List[Dict]:
    """Return a copy of the default rule base."""
    import copy
    return copy.deepcopy(DEFAULT_RULES)


def rules_to_text(rules: List[Dict]) -> str:
    """Convert rules to human-readable text."""
    lines = []
    for i, rule in enumerate(rules, 1):
        ants = " AND ".join(
            f"{var} IS {term}" for var, term in rule["antecedents"]
        )
        con_var, con_term = rule["consequent"]
        weight = rule.get("weight", 1.0)
        lines.append(
            f"Rule {i}: IF {ants} THEN {con_var} IS {con_term} "
            f"(weight={weight:.1f})"
        )
    return "\n".join(lines)
