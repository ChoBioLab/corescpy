#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

from scipy.stats import median_abs_deviation
import numpy as np


def is_outlier(data, column, nmads: int):
    """Determine outliers [below, above median]."""
    # From SC Best Practices
    metric = data[column]
    if isinstance(nmads, (int, float)):
        nmads = [nmads, nmads]
    if nmads[0] is not None and nmads[1] is not None:
        outlier = (metric < np.median(metric) - nmads[
            0] * median_abs_deviation(metric)) | (np.median(
                metric) + nmads[1] * median_abs_deviation(metric) < metric)
    elif nmads[0] is None:
        outlier = (np.median(metric) + nmads[1] * median_abs_deviation(
                metric) < metric)
    elif nmads[1] is None:
        outlier = (metric < np.median(metric) - nmads[
            0] * median_abs_deviation(metric))
    else:
        raise ValueError("Can't have None for both `nmads` elements.")
    return outlier
