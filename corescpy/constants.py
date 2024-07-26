#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Constants

@author: E. N. Aslinger
"""


def get_panel_constants(panel_id=None):
    """Get constants."""
    cons = {
        "TUQ97N": dict(
            col_sample_id_o="sample_id",
            col_sample_id="Sample",
            col_subject="subject_id",
            col_slide="slide_id",
            col_inflamed="inflammation",
            col_stricture="stricture",
            col_condition="Condition",
            key_inflamed="inflamed",
            key_uninflamed="uninflamed",
            key_stricture="stricture",
            col_data_dir="description",
            col_object="out_file",
            col_tangram="tangram_prediction",
            col_segment="segmentation"
        ),
        "XR4UZH": dict(
            col_sample_id_o="sampleID",
            col_sample_id="Sample",
            col_subject="sourceID",
            col_slide="slide_ID",
            col_inflamed="inflamed",
            col_condition="disease_status",
            key_inflamed="inflamed",
            key_uninflamed="uninflamed",
            col_data_dir="file_path",
            col_object="out_file",
            col_tangram="tangram_prediction"
        )
    }
    if panel_id is not None:
        cons = cons[panel_id]
    return cons
