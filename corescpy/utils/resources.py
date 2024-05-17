import decoupler as dc
import requests
import pandas as pd
import numpy as np


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


# def get_topp_gene(genes, categories=None, p_threshold=0.05, max_results=20,
#                   min_genes=1, max_genes=1000, correction="FDR", **kwargs):
#     url = "https://toppgene.cchmc.org/API/"
#     if categories is None:
#         categories = ["Coexpression", "CoexpressionAtlas", "ToppGene"]
#     payload = {"Symbols": genes}
#     headers = {"Content-Type": "application/json"}
#     response = requests.post(f"{url}lookup", json=payload, headers=headers)
#     ensembl = response.json()
#     ensembl = [{"Entrez": g["Entrez"], "Symbol": g["OfficialSymbol"]}
#                for g in ensembl["Genes"]]
#     call = [dict(Category=c, PValue=p_threshold, MaxResults=max_results,
#                  MinGenes=min_genes, MaxGenes=max_genes, Genes=ensembl,
#                  Correction=correction) for c in categories]
#     resp = requests.post(f"{url}enrich", json={"Annotations": [call]},
#                          headers=headers)
#     if resp.status_code == 200:
#         return resp.json()
#     else:
#         print(f"Error: {resp.status_code}")
#         return None
#     return resp


def get_topp_gene(genes, no_return=False, verbose=True,
                  sources=None, name_pattern=None,
                  symbols=True, categories="ToppCell", max_results=1000):
    """Get ToppGene results."""
    url = "https://toppgene.cchmc.org/API/enrich"
    url_lu = "https://toppgene.cchmc.org/API/lookup"
    head = {"Content-Type": "application/json"}
    if isinstance(categories, str):
        categories = [categories]
    if isinstance(sources, str):
        sources = [sources]
    if symbols is True:
        response = requests.post(url_lu, json={"Symbols": genes},
                                 headers={"Content-Type": "application/json"})
        data = response.json()
        genes = [int(r["Entrez"]) for r in data["Genes"]]
    # params = {"Genes": genes, "categories": [
    #     {"Type": "ToppGene", "correction": "FDR"}]}
    # params = {"Genes": genes, "categories": "ToppCell", "correction": "FDR"}
    # params = {"Genes": genes, "categories": "ToppCell"}
    params = {"Genes": genes, "MaxResults": max_results}
    # print(f"requests.post('{url}', json={params}, headers={head})")
    response = requests.post(url, json=params, headers=head)
    try:
        results = response.json()
    except Exception as e:
        print(e)
        print(response.text)
        # return response
    dff = pd.DataFrame(results)
    dff = dff.Annotations.apply(lambda x: x if x[
        "Category"] in categories else np.nan).dropna().apply(pd.Series)[
            ["Name", "Source", "Genes", "GenesInTerm",
             "GenesInQuery", "GenesInTermInQuery", "TotalGenes", "PValue",
             "QValueFDRBH", "QValueFDRBY", "QValueBonferroni"]]
    dff.loc[:, "Genes"] = dff.Genes.apply(lambda x: [i["Symbol"] for i in x])
    if sources:
        dff = dff[dff.Source.isin(sources)]
    if name_pattern and dff.shape[0] > 0:
        if not isinstance(name_pattern, dict):  # if no per-source pattern...
            name_pattern = dict(zip(name_pattern, [name_pattern] * len(
                dff.Name.unique())))  # ...use same pattern for all
        dff.loc[:, "Keep"] = True  # start off assuming keeping all
        for x in name_pattern:  # for each source, see if pattern in name
            dff.loc[dff.Source == x, "Keep"] = dff.loc[
                dff.Source == x, "Name"].apply(lambda y: name_pattern[x] in y)
        dff = dff[dff.Keep].drop("Keep", axis=1)
    if verbose is True:
        print(dff.head())
    if no_return is False:
        return dff
