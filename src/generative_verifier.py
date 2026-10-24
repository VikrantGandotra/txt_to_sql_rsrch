"""Compatibility import for the retired generative selector.

New code should import select_with_dspy from src.dspy_verifier directly.
This compatibility entry point also uses DSPy ratings and the compiled judge.
"""

import warnings

from src.dspy_verifier import select_with_dspy


def select_with_generative(*args, **kwargs):
    warnings.warn(
        "select_with_generative is retired; use select_with_dspy (DSPy validity ratings)",
        DeprecationWarning, stacklevel=2,
    )
    return select_with_dspy(*args, **kwargs)
