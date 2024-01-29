import decoupler as dc


def get_markers_database(resource="PanglaoDB", organism="human",
                         canonical_only=True, **kwargs):
    """Get database of cell type marker genes."""
    try:
        markers = dc.get_resource("PanglaoDB", organism, **kwargs)
    except Exception:
        dc.show_resources()
    if canonical_only is True:
        markers = markers[markers["canonical_marker"].astype(str) == "True"]
    markers = markers[~markers.duplicated(["cell_type", "genesymbol"])]
    return markers
