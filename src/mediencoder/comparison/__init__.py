"""Paired, fixed-mechanism natural-effect comparisons."""

from .dgp import ComparisonConfig, draw_mechanism, load_mechanism, sample_data, save_mechanism
from .reporting import summarize, markdown_table

__all__ = ["ComparisonConfig", "draw_mechanism", "load_mechanism", "sample_data",
           "save_mechanism", "summarize", "markdown_table"]
