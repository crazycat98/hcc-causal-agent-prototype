"""Causal feature selection boundary for M1.

M1 deliberately simulates the result of FCI by returning a fixed candidate set.
This is documented in README and docs/DECISIONS.md so the prototype does not
overclaim causal discovery validity.
"""

from __future__ import annotations

from .features import CAUSAL_FEATURES


def simulate_fci_selection(all_feature_names: list[str]) -> dict[str, object]:
    missing = [name for name in CAUSAL_FEATURES if name not in all_feature_names]
    if missing:
        raise ValueError(f"Cannot select causal candidates; missing features: {missing}")

    return {
        "method": "simulated_fci_fixed_causal_candidates",
        "selected_features": list(CAUSAL_FEATURES),
        "notes": (
            "Prototype simplification: this list simulates FCI-selected causal "
            "candidate variables over synthetic data."
        ),
    }

