# this makes me legit mad. what the fuck?

import os

# Set number of threads to use for OpenBLAS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Set number of threads to use for MKL.
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import sys
from collections.abc import Sequence
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import cpu_count
from pathlib import Path
from shutil import copyfile
from typing import Annotated, TextIO

import typer
from anndata import read_h5ad
from arboreto.algo import genie3, grnboost2
from arboreto.utils import load_tf_names
from ctxcore.rnkdb import RankingDatabase, opendb
from dask.diagnostics import ProgressBar

from pyscenic.aucell import aucell
from pyscenic.export import add_scenic_metadata
from pyscenic.prune import _prepare_client, find_features, prune2df
from pyscenic.utils import add_correlation, modules_from_adjacencies

from .utils import (
    load_adjacencies,
    load_exp_matrix,
    load_modules,
    load_signatures,
    save_enriched_motifs,
    save_matrix,
    suffixes_to_separator,
)

try:
    if isinstance(__package__, str):
        VERSION = version(__package__)
    else:
        VERSION = "unknown"
except PackageNotFoundError:  # pragma: no cover
    VERSION = "unknown"

LOGGER = logging.getLogger(__name__)


class GRNMethod(StrEnum):
    GENIE3 = "genie3"
    GRNBOOST2 = "grnboost2"


class ComputingMethod(StrEnum):
    CUSTOM_MULTIPROCESSING = "custom_multiprocessing"
    DASK_MULTIPROCESSING = "dask_multiprocessing"
    DASK_CLUSTER = "dask_cluster"


scenic = typer.Typer(
    name="pyscenic",
    help="Single-Cell rEgulatory Network Inference and Clustering (SCENIC), a tool for gene regulatory network inference and analysis.",
    add_completion=False,
    no_args_is_help=True,
    add_help_option=True,
)


@scenic.command(
    name="grn",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def find_adjacencies_command(
    expression_mtx_fname: Annotated[
        Path,
        typer.Argument(
            help=(
                "The name of the file that contains the expression matrix for the single cell experiment. CSV"
                "is supported (rows=cells x columns=genes)."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    tfs_fname: Annotated[
        Path,
        typer.Argument(
            help="The name of the file that contains the list of transcription factors (TXT; one TF per line).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output file path.")],
    # this originally stored a "yes" or "no", so ensure that we change any code that used it to work on an bool instead
    transpose: Annotated[
        bool,
        typer.Option(
            "--transpose",
            "-t",
            help="Transpose the expression matrix (rows=genes x columns=cells).",
        ),
    ] = False,
    method: Annotated[
        GRNMethod,
        typer.Option(
            "--method",
            "-m",
            help="The algorithm for gene regulatory network reconstruction.",
        ),
    ] = GRNMethod.GRNBOOST2,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Seed value for regressor random state initialization."),
    ] = None,
    num_workers: Annotated[
        int | None,
        typer.Option(
            "--num_workers",
            help="The number of workers to use. Only valid if using dask_multiprocessing, custom_multiprocessing or local as mode.",
        ),
    ] = None,
    client_or_address: Annotated[
        str,
        typer.Option(
            "--client_or_address",
            help="The client or the IP address of the dask scheduler to use."
            " (Only required of dask_cluster is selected as mode)",
        ),
    ] = "local",
    sparse: Annotated[
        bool,
        typer.Option(
            "--sparse",
            help="If set, load the expression data as a sparse matrix. Currently applies to the grn inference step only.",
        ),
    ] = False,
):
    """Infer co-expression modules."""
    if num_workers is None:
        num_workers = cpu_count()

    LOGGER.info("Loading expression matrix.")
    try:
        ex_mtx = load_exp_matrix(
            fname=expression_mtx_fname,
            transpose=transpose,
            return_sparse=sparse,
        )
    except ValueError as e:
        LOGGER.error(e)
        sys.exit(1)

    tf_names = load_tf_names(tfs_fname.name)

    if sparse:
        n_total_genes = len(ex_mtx[1])
        n_matching_genes = len(ex_mtx[1].isin(tf_names))
    else:
        n_total_genes = len(ex_mtx.columns)
        n_matching_genes = len(ex_mtx.columns.isin(tf_names))
    if n_total_genes == 0:
        LOGGER.error(
            "The expression matrix supplied does not contain any genes. "
            "Make sure the extension of the file matches the format (tab separation for TSV and "
            "comma sepatration for CSV)."
        )
        sys.exit(1)
    if float(n_matching_genes) / n_total_genes < 0.80:
        LOGGER.warning("Expression data is available for less than 80% of the supplied transcription factors.")

    LOGGER.info("Inferring regulatory networks.")
    client, shutdown_callback = _prepare_client(client_or_address=client_or_address, num_workers=num_workers)
    method = grnboost2 if method == "grnboost2" else genie3
    try:
        if sparse:
            network = method(
                expression_data=ex_mtx[0],
                gene_names=ex_mtx[1],
                tf_names=tf_names,
                verbose=True,
                client_or_address=client,
                seed=seed,
            )
        else:
            network = method(
                expression_data=ex_mtx,
                tf_names=tf_names,
                verbose=True,
                client_or_address=client,
                seed=seed,
            )
    finally:
        shutdown_callback(False)

    LOGGER.info("Writing results to file.")
    network.to_csv(output, index=False, sep=suffixes_to_separator(output.suffix))


def adjacencies2modules(
    module_fname: Path,
    expression_mtx_fname: Path,
    transpose: bool,
    thresholds: Sequence[float],
    top_n_targets: Sequence[int],
    top_n_regulators: Sequence[int],
    min_genes: int,
    mask_dropouts: bool,
    all_modules: bool = True,
):
    try:
        adjacencies = load_adjacencies(module_fname)
    except ValueError as e:
        LOGGER.error(e)
        sys.exit(1)

    LOGGER.info("Loading expression matrix.")
    try:
        ex_mtx = load_exp_matrix(
            expression_mtx_fname,
            transpose,
            False,  # sparse loading is disabled here for now
        )
    except ValueError as e:
        LOGGER.error(e)
        sys.exit(1)

    return modules_from_adjacencies(
        adjacencies,
        ex_mtx,
        thresholds=thresholds,
        top_n_targets=top_n_targets,
        top_n_regulators=top_n_regulators,
        min_genes=min_genes,
        rho_mask_dropouts=mask_dropouts,
        keep_only_activating=all_modules,
    )


@scenic.command(
    name="add_cor",
    help=(
        "Add Pearson correlations based on TF-gene expression to the network adjacencies output from the GRN step, "
        "and output these to a new adjacencies file. This will normally be done during the 'ctx' step."
    ),
)
def addCorrelations(
    adjacencies: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the GRN adjacencies (output from the GRN step).",
        ),
    ],
    expression_mtx_fname: Annotated[
        Path,
        typer.Option(
            "--expression_mtx_fname",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the expression matrix for the single cell experiment. CSV format is supported (rows=cells x columns=genes)",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file/stream, i.e. the adjacencies table with correlations.",
        ),
    ] = None,
    transpose: Annotated[bool, typer.Option("--transpose", "-t", help="Transpose the expression matrix.")] = False,
    mask_dropouts: Annotated[
        bool,
        typer.Option(
            "--mask_dropouts",
            help=(
                "If modules need to be generated, this controls whether cell dropouts (cells in which expression of "
                "either TF or target gene is 0) are masked when calculating the correlation between a TF-target pair. "
                "This affects which target genes are included in the initial modules, and the final pruned regulon (by "
                "default only positive regulons are kept (see --all_modules option)). The default value in pySCENIC "
                "0.9.16 and previous versions was to mask dropouts when calculating the correlation; however, all cells "
                "are now kept by default, to match the R version."
            ),
        ),
    ] = False,
):
    output = output if output is not None else sys.stdout
    adjacencies = load_adjacencies(adjacencies)

    LOGGER.info("Loading expression matrix.")
    ex_mtx = load_exp_matrix(
        fname=expression_mtx_fname,
        transpose=transpose,
        return_sparse=False,  # sparse loading is disabled here for now
    )

    LOGGER.info("Calculating correlations.")
    adjacencies_wCor = add_correlation(adjacencies, ex_mtx, rho_threshold=0.03, mask_dropouts=mask_dropouts)

    LOGGER.info("Writing results to file.")
    if isinstance(output, TextIO):
        adjacencies_wCor.to_csv(output, index=False, sep=",")
    else:
        adjacencies_wCor.to_csv(output, index=False, sep=suffixes_to_separator(output.suffix))


def _load_dbs(fnames: Sequence[Path]) -> Sequence[type[RankingDatabase]]:
    return [opendb(fname=fname, name=fname.name) for fname in fnames]


class NoProgressBar:
    def __enter__(self):
        return self

    def __exit__(*x):
        pass


@scenic.command(
    name="ctx",
    help="Find enriched motifs for a gene signature and optionally prune targets from this signature based on cis-regulatory cues.",
)
def prune_targets_command(
    module_fname: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the signature or the co-expression modules. "
            "The following formats are supported: CSV or TSV (adjacencies), YAML, GMT and DAT (modules)",
        ),
    ],
    database_fname: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name(s) of the regulatory feature databases. Two file formats are supported: feather or db (legacy).",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file/stream, i.e. a table of enriched motifs and target genes (csv, tsv) or collection of regulons (yaml, gmt, dat, json).",
        ),
    ],
    no_pruning: Annotated[
        bool,
        typer.Option(
            "--no_pruning",
            "-n",
            help="Do not perform pruning, i.e. find enriched motifs.",
        ),
    ],
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk_size",
            "-c",
            help="The size of the module chunks assigned to a node in the dask graph.",
        ),
    ] = 100,
    mode: Annotated[
        ComputingMethod,
        typer.Option("--mode", help="The mode to be used for computing."),
    ] = ComputingMethod.CUSTOM_MULTIPROCESSING,
    all_modules: Annotated[
        bool,
        typer.Option(
            "--all_modules",
            "-a",
            help="Included positive and negative regulons in the analysis (default: no, i.e. only positive).",
        ),
    ] = False,
    transpose: Annotated[bool, typer.Option("--transpose", "-t", help="Transpose the expression matrix.")] = False,
    rank_threshold: Annotated[
        int,
        typer.Option(
            "--rank_threshold",
            help="The rank threshold used for deriving the target genes of an enriched motif.",
        ),
    ] = 5000,
    auc_threshold: Annotated[
        float,
        typer.Option(
            "--auc_threshold",
            help="The threshold used for calculating the AUC of a feature as fraction of ranked genes.",
        ),
    ] = 0.05,
    nes_threshold: Annotated[
        float,
        typer.Option(
            "--nes_threshold",
            help="The Normalized Enrichment Score (NES) threshold for finding enriched features.",
        ),
    ] = 3.0,
    min_orthologous_identity: Annotated[
        float,
        typer.Option(
            "--min_orthologous_identity",
            help="Minimum orthologous identity to use when annotating enriched motifs.",
        ),
    ] = 0.0,
    max_similarity_fdr: Annotated[
        float,
        typer.Option(
            "--max_similarity_fdr",
            help="Maximum FDR in motif similarity to use when annotating enriched motifs.",
        ),
    ] = 0.001,
    annotations_fname: Annotated[
        Path | None,
        typer.Option(
            "--annotations_fname",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the motif annotations to use.",
        ),
    ] = None,
    num_workers: Annotated[
        int | None,
        typer.Option(
            "--num_workers",
            help="The number of workers to use. Only valid if using dask_multiprocessing, custom_multiprocessing or local as mode.",
        ),
    ] = None,
    # TODO make this an enum
    client_or_address: Annotated[
        str,
        typer.Option(
            "--client_or_address",
            help="The client or the IP address of the dask scheduler to use."
            " (Only required of dask_cluster is selected as mode)",
        ),
    ] = "local",
    thresholds: Annotated[
        list[float],
        typer.Option(
            "--thresholds",
            help="The first method to create the TF-modules based on the best targets for each transcription factor.",
        ),
    ] = [0.75, 0.90],
    top_n_targets: Annotated[
        list[int],
        typer.Option(
            "--top_n_targets",
            help="The second method is to select the top targets for a given TF.",
        ),
    ] = [50],
    top_n_regulators: Annotated[
        list[int],
        typer.Option(
            "--top_n_regulators",
            help="The alternative way to create the TF-modules is to select the best regulators for each gene.",
        ),
    ] = [5, 10, 50],
    min_genes: Annotated[
        int,
        typer.Option(
            "--min_genes",
            help="The minimum number of genes in a module.",
        ),
    ] = 20,
    mask_dropouts: Annotated[
        bool,
        typer.Option(
            "--mask_dropouts",
            help="If modules need to be generated, this controls whether cell dropouts (cells in which expression of either TF or target gene is 0) are masked when calculating the correlation between a TF-target pair. This affects which target genes are included in the initial modules, and the final pruned regulon (by default only positive regulons are kept (see --all_modules option)). The default value in pySCENIC 0.9.16 and previous versions was to mask dropouts when calculating the correlation; however, all cells are now kept by default, to match the R version.",
        ),
    ] = False,
    expression_mtx_fname: Annotated[
        Path | None,
        typer.Option(
            "--expression_mtx_fname",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the expression matrix for the single cell experiment. CSV format is supported (rows=cells x columns=genes)",
        ),
    ] = None,
):
    """Prune targets/find enriched features."""
    # Loading from YAML is extremely slow. Therefore this is a potential performance improvement.
    # Potential improvements are switching to JSON or to use a CLoader:
    # https://stackoverflow.com/questions/27743711/can-i-speedup-yaml
    # The alternative for which was opted in the end is binary pickling.

    # TODO can we just replace that with with msgspec? Would love to avoid pickling.
    if module_fname.suffix in [".csv", ".tsv"]:
        if expression_mtx_fname is None:
            LOGGER.error("No expression matrix is supplied.")
            sys.exit(0)
        LOGGER.info("Creating modules.")
        modules = adjacencies2modules(
            module_fname,
            expression_mtx_fname,
            transpose,
            thresholds,
            top_n_targets,
            top_n_regulators,
            min_genes,
            mask_dropouts,
            all_modules,
        )
    else:
        LOGGER.info("Loading modules.")
        try:
            modules = load_modules(module_fname)
        except ValueError as e:
            LOGGER.error(e)
            sys.exit(1)

    if len(modules) == 0:
        LOGGER.error("Not a single module loaded")
        sys.exit(1)

    LOGGER.info("Loading databases.")
    dbs = _load_dbs(database_fname)

    LOGGER.info("Calculating regulons.")
    motif_annotations_fname = annotations_fname
    calc_func = find_features if no_pruning == "yes" else prune2df
    if mode == "dask_cluster":
        if not client_or_address:
            LOGGER.error('--mode "dask_cluster" requires --client_or_address argument.')
            sys.exit(1)
        else:
            client_or_mode = client_or_address
    else:
        client_or_mode = mode
    with ProgressBar() if mode == "dask_multiprocessing" else NoProgressBar():
        df_motifs = calc_func(
            dbs,
            modules,
            motif_annotations_fname,
            rank_threshold=rank_threshold,
            auc_threshold=auc_threshold,
            nes_threshold=nes_threshold,
            client_or_address=client_or_mode,
            module_chunksize=chunk_size,
            num_workers=num_workers,
            motif_similarity_fdr=max_similarity_fdr,
            orthologuous_identity_threshold=min_orthologous_identity,
        )

    LOGGER.info("Writing results to file.")
    if output.name == "<stdout>":
        df_motifs.to_csv(output)
    else:
        save_enriched_motifs(df_motifs, output)


@scenic.command(name="aucell", help="Quantify activity of gene signatures across single cells.")
def aucell_command(
    expression_mtx_fname: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the expression matrix for the single cell experiment. csv format is supported: (rows=cells x columns=genes).",
        ),
    ],
    signatures_fname: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="The name of the file that contains the gene signatures. Three file formats are supported: gmt, yaml or dat.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output file/stream, i.e. the AUC matrix (csv, tsv)."),
    ],
    transpose: Annotated[
        bool,
        typer.Option(
            "--transpose",
            help="Transpose the expression matrix (rows=genes x columns=cells).",
        ),
    ] = False,
    weights: Annotated[
        bool,
        typer.Option(
            "--weights",
            help="Whether to use weights for the AUC calculation (default: no).",
        ),
    ] = False,
    num_workers: Annotated[
        int,
        typer.Option(
            "--num_workers",
            help="The number of workers to use for parallel processing (default: 1).",
        ),
    ] = 1,
    seed: Annotated[
        int,
        typer.Option("--seed", help="The random seed for reproducibility (default: 42)."),
    ] = 42,
    auc_threshold: Annotated[
        float,
        typer.Option(
            "--auc_threshold",
            help="The threshold used for calculating the AUC of a feature as fraction of ranked genes.",
        ),
    ] = 0.05,
):
    """Calculate regulon enrichment (as AUC values) for cells."""
    LOGGER.info("Loading expression matrix.")
    try:
        ex_mtx = load_exp_matrix(
            expression_mtx_fname,
            transpose,
            False,  # sparse loading is disabled here for now
        )
    except ValueError as e:
        LOGGER.error(e)
        sys.exit(1)

    LOGGER.info("Loading gene signatures.")
    try:
        signatures = load_signatures(signatures_fname)
    except ValueError as e:
        LOGGER.error(e)
        sys.exit(1)

    LOGGER.info("Calculating cellular enrichment.")
    auc_mtx = aucell(
        ex_mtx,
        signatures,
        auc_threshold=auc_threshold,
        noweights=(not weights),
        seed=seed,
        num_workers=num_workers,
    )

    LOGGER.info("Writing results to file.")
    if output.suffix == ".h5ad":
        # check input file is also h5ad:
        if expression_mtx_fname.suffix == ".h5ad":
            copyfile(expression_mtx_fname, output)
            add_scenic_metadata(read_h5ad(filename=output, backed="r"), auc_mtx, signatures).write(output)
        else:
            LOGGER.error("Expression matrix should be provided in the h5ad (anndata) file format.")
            sys.exit(1)
    elif output == "<stdout>":
        (auc_mtx.T if transpose else auc_mtx).to_csv(output)
    else:
        save_matrix(auc_mtx, output, transpose)
