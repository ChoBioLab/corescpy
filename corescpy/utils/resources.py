import decoupler as dc
import requests
import json
import os


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


def get_topp_gene(genes, categories=None, p_threshold=0.05, max_results=20,
                  min_genes=1, max_genes=1000, correction="FDR", **kwargs):
    url = "https://toppgene.cchmc.org/API/"
    if categories is None:
        categories = ["Coexpression", "CoexpressionAtlas", "ToppGene"]
    payload = {"Symbols": genes}
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{url}lookup", json=payload, headers=headers)
    ensembl = response.json()
    ensembl = [{"Entrez": g["Entrez"], "Symbol": g["OfficialSymbol"]}
               for g in ensembl["Genes"]]
    # call_cat = [dict(Type=c, PValue=p_threshold, MaxResults=max_results,
    #                  MinGenes=min_genes, MaxGenes=max_genes,
    #                  Correction=correction) for c in categories]
    # call = json.dumps(dict(Genes=genes, Categories=call_cat),
    #                   ensure_ascii=False)
    # call = str(
    #     f"curl -H 'Content-Type: application/json' -d '{call}'"
    #     " https://toppgene.cchmc.org/API/enrich")
    # os.system(call)
    # call = dict(Genes=[{"Entrez": g["Entrez"], "Symbol": g["OfficialSymbol"]}
    #                    for g in ensembl["Genes"]], Categories=call_cat)
    # call = dict(Genes=[{"Entrez": g["Entrez"], "Symbol": g["OfficialSymbol"]}
    #                    for g in ensembl["Genes"]], Categories=call_cat)
    call = [dict(Category=c, PValue=p_threshold, MaxResults=max_results,
                 MinGenes=min_genes, MaxGenes=max_genes, Genes=ensembl,
                 Correction=correction) for c in categories]
    resp = requests.post(f"{url}enrich", json={"Annotations": [call]},
                         headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Error: {resp.status_code}")
        return None
    return resp
