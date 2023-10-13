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
    """Determine outliers (from SC Best Practices)."""
    metric = data[column]
    outlier = (metric < np.median(metric) - nmads * median_abs_deviation(
        metric)) | (np.median(metric) + nmads * median_abs_deviation(
            metric) < metric)
    return outlier