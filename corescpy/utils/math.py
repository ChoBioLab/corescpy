#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

from scipy.stats import median_abs_deviation
import numpy as np


def is_outlier(data, column, nmads):
    """Determine outliers [below, above median]."""
    # From SC Best Practices
    metric = data[column]
    if isinstance(nmads, (int, float)):
        nmads = [nmads, nmads]
    mad = median_abs_deviation(metric)
    thresh = [np.median(metric) - nmads[0] * mad,
              np.median(metric) + nmads[1] * mad]  # minimum, maximum
    if nmads[0] is not None and nmads[1] is not None:
        outlier = (metric < thresh[0]) | (thresh[1] < metric)
    elif nmads[0] is None:  # not calculating based on a minimum
        thresh[0] = None  # no minimum
        outlier = (np.median(metric) + nmads[1] * mad < metric)
        outlier = (metric < thresh[0]) | (thresh[1] < metric)
    elif nmads[1] is None:  # not calculating based on a maximum
        thresh[1] = None  # no maximum
        outlier = metric < thresh[0]
    else:
        raise ValueError("Can't have None for both `nmads` elements.")
    return outlier, thresh  # y/n, threshold
