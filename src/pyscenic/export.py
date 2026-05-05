from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from ctxcore.genesig import Regulon

from .binarization import binarize


def add_scenic_metadata(
    adata: AnnData,
    auc_mtx: pd.DataFrame,
    regulons: Sequence[Regulon] | None = None,
    bin_rep: bool = False,
    copy: bool = False,
) -> AnnData:
    """
    Add AUCell values and regulon metadata to AnnData object.
    :param adata: The AnnData object.
    :param auc_mtx: The dataframe containing the AUCell values (#observations x #regulons).
    :param bin_rep: Also add binarized version of AUCell values as separate representation. This representation
    is stored as `adata.obsm['X_aucell_bin']`.
    :param copy: Return a copy instead of writing to adata.
    :
    """
    # To avoid dependency with scanpy package the type hinting intentionally uses string literals.
    # In addition, the assert statement to assess runtime type is also commented out.
    # assert isinstance(adata, sc.AnnData)
    assert isinstance(auc_mtx, pd.DataFrame)
    assert len(auc_mtx) == adata.n_obs

    REGULON_SUFFIX_PATTERN = "Regulon({})"

    result = adata.copy() if copy else adata

    # Add AUCell values as new representation (similar to a PCA). This facilitates the usage of
    # AUCell as initial dimensional reduction.
    result.obsm["X_aucell"] = auc_mtx.values.copy()
    if bin_rep:
        bin_mtx, _ = binarize(auc_mtx)
        result.obsm["X_aucell_bin"] = bin_mtx.values

    # Encode genes in regulons as "binary" membership matrix.
    if regulons is not None:
        genes = np.array(adata.var_names)
        data = np.zeros(shape=(adata.n_vars, len(regulons)), dtype=bool)
        for idx, regulon in enumerate(regulons):
            data[:, idx] = np.isin(genes, regulon.genes).astype(bool)
        regulon_assignment = pd.DataFrame(
            data=data,
            index=genes,
            columns=[REGULON_SUFFIX_PATTERN.format(r.name) for r in regulons],
        )
        result.var = pd.merge(
            result.var,
            regulon_assignment,
            left_index=True,
            right_index=True,
            how="left",
        )

    # Add additional meta-data/information on the regulons.
    def fetch_logo(context):
        for elem in context:
            if elem.endswith(".png"):
                return elem
        return ""

    result.uns["aucell"] = {
        "regulon_names": auc_mtx.columns.map(REGULON_SUFFIX_PATTERN.format).values,
        "regulon_motifs": np.array([fetch_logo(reg.context) for reg in regulons] if regulons is not None else []),
    }

    # Add the AUCell values also as annotations of observations. This way regulon activity can be
    # depicted on cellular scatterplots.
    mtx = auc_mtx.copy()
    mtx.columns = result.uns["aucell"]["regulon_names"]
    result.obs = pd.merge(result.obs, mtx, left_index=True, right_index=True, how="left")

    return result


def export_regulons(regulons: Sequence[Regulon], fname: str) -> None:
    """
    Export regulons as GraphML.
    :param regulons: The sequence of regulons to export.
    :param fname: The name of the file to create.
    """
    graph = nx.DiGraph()
    for regulon in regulons:
        src_name = regulon.transcription_factor
        graph.add_node(src_name, group="transcription_factor")
        edge_type = "activating" if "activating" in regulon.context else "inhibiting"
        node_type = "activated_target" if "activating" in regulon.context else "inhibited_target"
        for dst_name, edge_strength in regulon.gene2weight.items():
            graph.add_node(dst_name, group=node_type, **regulon.context)
            graph.add_edge(
                src_name,
                dst_name,
                weight=edge_strength,
                interaction=edge_type,
                **regulon.context,
            )
    nx.readwrite.write_graphml(graph, fname)
