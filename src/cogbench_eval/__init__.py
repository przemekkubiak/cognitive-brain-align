"""
CogBench Model Evaluation Suite

"""

__version__ = "0.1.0"
__author__ = "Przemek Kubiak"

from . import analysis  # noqa: F401
from . import evaluation  # noqa: F401

__all__ = ["evaluation", "analysis"]
