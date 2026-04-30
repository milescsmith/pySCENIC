# -*- coding: utf-8 -*-

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Sequence, Type

import numpy as np
import pandas as pd
from ctxcore.genesig import GeneSignature, openfile

# from pyscenic.binarization import binarize
from pyscenic.transform import df2regulons
from pyscenic.utils import load_from_yaml, load_motifs, save_to_yaml

__all__ = [
    "save_matrix",
    "load_exp_matrix",
    "load_signatures",
    "save_enriched_motifs",
    "load_adjacencies",
    "load_modules",
]


ATTRIBUTE_NAME_CELL_IDENTIFIER = "CellID"
ATTRIBUTE_NAME_GENE = "Gene"
ATTRIBUTE_NAME_REGULONS_AUC = "RegulonsAUC"
ATTRIBUTE_NAME_REGULONS = "Regulons"
ATTRIBUTE_NAME_METADATA = "MetaData"


def suffixes_to_separator(extension):
    if ".csv" in extension:
        return ","
    if ".tsv" in extension:
        return "\t"


def is_valid_suffix(extension, method):
    if not isinstance(extension, Sequence):
        msg = 'extension should be of type "list"'
        raise ValueError(msg)

    if method in ["grn", "aucell"]:
        valid_extensions = [".csv", ".tsv", ".h5ad"]
    elif method == "ctx":
        valid_extensions = [".csv", ".tsv"]
    elif method == "ctx_yaml":
        valid_extensions = [".yaml", ".yml"]
    if len(set(extension).intersection(valid_extensions)) > 0:
        return True
    else:
        return False


def load_exp_matrix(
    fname: Path,
    transpose: bool = False,
    return_sparse: bool = False,
) -> pd.DataFrame:
    """
    Load expression matrix from disk.

    Supported file formats are CSV, and TSV.

    :param fname: The name of the file that contains the expression matrix.
    :param transpose: Is the expression matrix stored as (rows = genes x columns = cells)?
    :param return_sparse: Return a sparse matrix
    :return: A 2-dimensional dataframe (rows = cells x columns = genes).
    """
    if fname.suffix in [".csv", ".tsv", ".h5ad"]:
        if fname.suffix == ".h5ad":
            from anndata import read_h5ad

            adata = read_h5ad(filename=fname, backed="r")
            if return_sparse:
                # expr, gene, cell:
                return adata.X.value, adata.var_names.values, adata.obs_names.values
            else:
                return pd.DataFrame(
                    adata.X.value.todense(),
                    index=adata.obs_names.values,
                    columns=adata.var_names.values,
                )

        else:
            df = pd.read_csv(
                fname, sep=suffixes_to_separator(fname.suffix), header=0, index_col=0
            )
            return df.T if transpose else df
    else:
        msg = f'Unknown file format "{fname!s}".'
        raise ValueError(msg)


def save_matrix(df: pd.DataFrame, fname: Path, transpose: bool = False) -> None:
    """
    Save matrix to disk.

    Supported file formats are CSV, and TSV.

    :param df: A 2-dimensional dataframe (rows = cells x columns = genes).
    :param fname: The name of the file to be written.
    :param transpose: Should the expression matrix be stored as (rows = genes x columns = cells)?
    """
    if fname.suffix in [".csv", ".tsv", ".h5ad"]:
        (df.T if transpose else df).to_csv(
            fname, sep=suffixes_to_separator(fname.suffix)
        )
    else:
        msg = f'Unknown file format "{fname!s}".'
        raise ValueError(msg)


def guess_separator(fname: str) -> str:
    with openfile(fname, "r") as f:
        lines = f.readlines()

    # decode if gzipped file:
    for i, x in enumerate(lines):
        if isinstance(x, (bytes, bytearray)):
            lines[i] = x.decode()

    def count_columns(sep):
        return [
            len(line.split(sep))
            for line in lines
            if not line.strip().startswith("#") and line.strip()
        ]

    # Check if '\t' is used:
    for sep in ("\t", ";", ","):
        if min(count_columns(sep)) >= 3:
            return sep
    raise ValueError('Unknown file format "{}".'.format(fname))


def load_signatures(fname: Path) -> Sequence[Type[GeneSignature]]:
    """
    Load genes signatures from disk.

    Supported file formats are GMT, DAT (pickled), YAML or CSV (enriched motifs).

    :param fname: The name of the file that contains the signatures.
    :return: A list of gene signatures.
    """
    extension = fname.suffix
    if is_valid_suffix(extension, "ctx"):
        # csv/tsv
        return df2regulons(load_motifs(fname, sep=suffixes_to_separator(extension)))
    elif is_valid_suffix(extension, "ctx_yaml"):
        return load_from_yaml(fname)
    elif ".gmt" in extension:
        sep = guess_separator(fname)
        return GeneSignature.from_gmt(fname, field_separator=sep, gene_separator=sep)
    elif ".dat" in extension:
        with openfile(fname, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError('Unknown file format "{}".'.format(fname))


def save_enriched_motifs(df, fname: Path) -> None:
    """
    Save enriched motifs.

    Supported file formats are CSV, TSV, GMT, DAT (pickle), JSON or YAML.

    :param df:
    :param fname:
    :return:
    """
    extension = fname.suffix
    if is_valid_suffix(extension, "ctx"):
        df.to_csv(fname, sep=suffixes_to_separator(extension))
    else:
        regulons = df2regulons(df)
        match extension:
            case ".json":
                name2targets = {r.name: list(r.gene2weight.keys()) for r in regulons}
                with openfile(fname, "w") as f:
                    f.write(json.dumps(name2targets))
            case ".dat":
                with openfile(fname, "wb") as f:
                    pickle.dump(regulons, f)
            case ".gmt":
                GeneSignature.to_gmt(fname, regulons)
            case _ if is_valid_suffix(extension, "ctx_yaml"):
                save_to_yaml(regulons, fname)
            case _:
                raise ValueError('Unknown file format "{}".'.format(fname))


def load_adjacencies(fname: Path) -> pd.DataFrame:
    return pd.read_csv(
        fname,
        sep=suffixes_to_separator(fname.suffix),
        dtype={0: str, 1: str, 2: np.float64},
        keep_default_na=False,
    )


def load_modules(fname: Path) -> Sequence[Type[GeneSignature]]:
    # Loading from YAML is extremely slow. Therefore this is a potential performance improvement.
    # Potential improvements are switching to JSON or to use a CLoader:
    # https://stackoverflow.com/questions/27743711/can-i-speedup-yaml
    # The alternative for which was opted in the end is binary pickling.
    extension = fname.suffixes
    if is_valid_suffix(extension, "ctx_yaml"):
        return load_from_yaml(fname)
    elif ".dat" in extension:
        with openfile(fname, "rb") as f:
            return pickle.load(f)
    elif ".gmt" in extension:
        return GeneSignature.from_gmt(fname)
    else:
        raise ValueError('Unknown file format for "{}".'.format(fname))


def decompress_meta(meta):
    try:
        meta = meta.decode("ascii")
        return json.loads(zlib.decompress(base64.b64decode(meta)))
    except AttributeError:
        return json.loads(
            zlib.decompress(base64.b64decode(meta.encode("ascii"))).decode("ascii")
        )


def compress_meta(meta):
    return base64.b64encode(zlib.compress(json.dumps(meta).encode("ascii"))).decode(
        "ascii"
    )
