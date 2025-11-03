import base64
import io
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from scipy.stats import ttest_ind
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from pathlib import Path

HERE = Path(__file__).resolve().parent
# If the app file lives in a subfolder, this will still find /static at the project root
CANDIDATES = [HERE / "static", HERE.parent / "static", HERE.parent.parent / "static"]
STATIC_DIR = next((p for p in CANDIDATES if p.exists()), HERE / "static")

CANDIDATES_T = [
    HERE / "templates",
    HERE.parent / "templates",
    HERE.parent.parent / "templates",
]
TEMPLATE_DIR = next((p for p in CANDIDATES_T if p.exists()), HERE / "templates")

app = Flask(
    __name__,
    static_folder=str(HERE / "static"),
    static_url_path="/static",
    template_folder=str(HERE / "templates"),
)

matplotlib.use("Agg")

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).expanduser()
COUNT_PATH = DATA_DIR / "count_data.parquet"
ANNO_PATH = DATA_DIR / "anno_data.parquet"
MPLCONFIG_DIR = Path(os.getenv("MPLCONFIGDIR", DATA_DIR / ".mpl-cache")).expanduser()
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
COUNT_FILE_ID = os.getenv("COUNT_FILE_ID")
ANNO_FILE_ID = os.getenv("ANNO_FILE_ID")
SA_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
logger = logging.getLogger(__name__)


def _drive_client():
    info = _load_service_account_info()
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def _ensure_local(path: Path, file_id: str):
    if path.exists() and path.stat().st_size > 0:
        return path
    if path.exists():
        path.unlink()
    normalized_id = _normalize_file_id(file_id)
    if not normalized_id:
        raise RuntimeError(f"File identifier for {path.name} is not configured.")
    service = _drive_client()
    request = service.files().get_media(fileId=normalized_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return path


def _normalize_file_id(file_ref: str | None) -> str:
    if not file_ref:
        return ""
    value = file_ref.strip()
    if "://" not in value:
        return value
    parsed = urlparse(value)
    if parsed.netloc.endswith("drive.google.com"):
        if "/file/d/" in parsed.path:
            segments = parsed.path.split("/")
            try:
                idx = segments.index("d")
                return segments[idx + 1]
            except (ValueError, IndexError):
                pass
        query = parse_qs(parsed.query)
        if "id" in query:
            return query["id"][0]
    raise RuntimeError(
        "GOOGLE Drive file reference must be a file ID or share link with an embedded ID."
    )


def _warm_resources():
    try:
        if COUNT_FILE_ID:
            _ensure_local(COUNT_PATH, COUNT_FILE_ID)
        if ANNO_FILE_ID:
            _ensure_local(ANNO_PATH, ANNO_FILE_ID)
    except Exception as exc:
        logger.warning("Unable to prefetch data: %s", exc)
    if os.getenv("WARM_MPL", "1") != "0":
        try:
            fig, ax = plt.subplots()
            ax.plot([0], [0])
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
        except Exception as exc:
            logger.warning("Unable to warm matplotlib cache: %s", exc)


if os.getenv("PRELOAD_DATA", "1") != "0":
    try:
        _warm_resources()
    except Exception as exc:
        logger.warning("Warm-up failed: %s", exc)


def _load_service_account_info() -> dict:
    if not SA_JSON:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is not configured for Drive access."
        )
    raw = SA_JSON.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    candidate = Path(raw).expanduser()
    if not candidate.exists():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON must be a JSON string or path to the credential file."
        )
    return json.loads(candidate.read_text())


@app.errorhandler(Exception)
def _handle_exception(error):
    if isinstance(error, HTTPException):
        return error
    logger.exception("Unhandled error")
    return jsonify(error="Internal server error"), 500


def _read_gene_row(gene: str) -> pd.DataFrame:
    pf = pq.ParquetFile(COUNT_PATH)
    gene_scalar = pa.scalar(gene)
    for rg in range(pf.num_row_groups):
        gene_only = pf.read_row_group(rg, columns=["Gene"])
        mask = pc.equal(gene_only.column("Gene"), gene_scalar)
        if pc.any(mask).as_py():
            full = pf.read_row_group(rg)
            filtered = full.filter(pc.equal(full.column("Gene"), gene_scalar))
            return filtered.to_pandas()
    return pd.DataFrame()


@lru_cache(maxsize=1)
def _load_annotation_base() -> pd.DataFrame:
    _ensure_local(ANNO_PATH, ANNO_FILE_ID)
    df = pd.read_parquet(
        ANNO_PATH,
        columns=[
            "type",
            "Age",
            "SegmentDisplayName",
            "Diagnosis_ralph",
            "Ki67 percentage",
            "p53 pattern",
            "Morphology category",
            "BRCA category",
            "Molecular_Subtype",
            "Decision Tree",
        ],
    )
    rename_dict = {
        "Normal fallopian tube epithelium": "NFT",
        "p53 signature": "p53 sig",
        "STIC (incidental)": "STIC",
        "STIC-like lesion": "STICL",
        "High grade serous carcinoma": "HGSC",
    }
    df["Diagnosis"] = df["Diagnosis_ralph"].replace(rename_dict)
    relevant_diagnoses = ["NFT", "p53 sig", "STIL", "STIC", "STICL", "HGSC"]
    df = df[df["Diagnosis"].isin(relevant_diagnoses)]
    df["Diagnosis"] = pd.Categorical(
        df["Diagnosis"], categories=relevant_diagnoses, ordered=True
    )
    df["Molecular_Subtype"] = pd.Categorical(
        df["Molecular_Subtype"],
        categories=[
            "NFT",
            "Dormant",
            "Mixed",
            "Immunoreactive",
            "Proliferative",
            "HGSC",
        ],
        ordered=True,
    )
    return df


def _prepare_annotation(sel_type: str | None) -> pd.DataFrame:
    data = _load_annotation_base()
    if sel_type:
        data = data[data["type"] == sel_type]
    return data


@app.route("/")
def home():
    return render_template("pre_cancer_atlas.html")


@app.route("/pre_cancer_atlas", methods=["POST"])
def plot():
    gene = request.form["gene"].strip().upper()

    # Load the count data for the specific gene
    try:
        _ensure_local(COUNT_PATH, COUNT_FILE_ID)
        gene_row = _read_gene_row(gene)
    except FileNotFoundError:
        return (
            jsonify(
                error=(
                    "Count data file not found. "
                    "Set DATA_DIR to the directory containing count_data.parquet."
                )
            ),
            500,
        )
    except Exception as exc:
        return (
            jsonify(error=f"Unable to load count data: {exc}"),
            500,
        )

    # Check if the gene exists
    if gene_row.empty:
        return jsonify(error=f"The gene '{gene}' is not in the database."), 404

    # Select data for the chosen gene across all samples
    gene_row = gene_row.set_index("Gene")
    gene_data = gene_row.loc[gene].to_frame(name="Count")
    gene_data.reset_index(inplace=True)
    gene_data.columns = ["Sample", "Count"]

    selection = request.form.get("selection", "").strip()

    # Load annotation data and apply renaming and filtering
    sel_type = request.form.get("type")
    sel_type = sel_type or None
    try:
        selected_annotations = _prepare_annotation(sel_type)
        stroma_annotations = _prepare_annotation("stroma")
    except FileNotFoundError:
        return (
            jsonify(
                error=(
                    "Annotation data file not found. "
                    "Set DATA_DIR to the directory containing anno_data.parquet."
                )
            ),
            500,
        )
    except Exception as exc:
        return (
            jsonify(error=f"Unable to load annotation data: {exc}"),
            500,
        )

    categories_order = [
        "Stroma",
        "NFT",
        "Dormant",
        "Mixed",
        "Immunoreactive",
        "Proliferative",
        "HGSC",
    ]

    def _merge_with_annotations(
        annotations: pd.DataFrame, force_label: str | None = None
    ) -> pd.DataFrame:
        merged = pd.merge(
            gene_data, annotations, left_on="Sample", right_on="SegmentDisplayName"
        )
        if merged.empty:
            return merged
        merged = merged.copy()
        if force_label is not None:
            merged["Molecular_Subtype"] = force_label
        merged["Molecular_Subtype"] = pd.Categorical(
            merged["Molecular_Subtype"], categories=categories_order, ordered=True
        )
        merged = merged.dropna(subset=["Molecular_Subtype"])
        return merged

    selected_data = _merge_with_annotations(selected_annotations)
    stroma_data = _merge_with_annotations(stroma_annotations, force_label="Stroma")
    plot_data = pd.concat([stroma_data, selected_data], ignore_index=True)
    plot_data["Molecular_Subtype"] = pd.Categorical(
        plot_data["Molecular_Subtype"], categories=categories_order, ordered=True
    )
    plot_data = plot_data.dropna(subset=["Molecular_Subtype"])

    if plot_data.empty:
        return jsonify(error="No annotation data available for plotting."), 404

    present_categories = [
        cat
        for cat in plot_data["Molecular_Subtype"].cat.categories
        if (plot_data["Molecular_Subtype"] == cat).any()
    ]
    category_positions = {cat: idx + 1 for idx, cat in enumerate(present_categories)}

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data.boxplot(
        column="Count",
        by="Molecular_Subtype",
        ax=ax,
        flierprops={"marker": ""},
    )

    jitter_strength = 0.06
    point_size = 5
    for subtype in present_categories:
        mask = plot_data["Molecular_Subtype"] == subtype
        x = np.random.normal(
            category_positions[subtype], jitter_strength, size=mask.sum()
        )
        y = plot_data.loc[mask, "Count"]
        ax.scatter(x, y, color="black", alpha=0.7, s=point_size)

    nft_counts = plot_data.loc[plot_data["Molecular_Subtype"] == "NFT", "Count"]
    if len(nft_counts) >= 2:
        for subtype in present_categories:
            if subtype in {"NFT", "Stroma"}:
                continue
            subtype_counts = plot_data.loc[
                plot_data["Molecular_Subtype"] == subtype, "Count"
            ]
            if len(subtype_counts) < 2:
                continue
            _, p_value = ttest_ind(nft_counts, subtype_counts, equal_var=False)
            marker = None
            if p_value < 0.001:
                marker = "***"
            elif p_value < 0.01:
                marker = "**"
            elif p_value < 0.05:
                marker = "*"
            if marker:
                y_max = plot_data["Count"].max()
                y_min = plot_data["Count"].min()
                padding = max((y_max - y_min) * 0.1, 0.1 * max(abs(y_max), 1))
                ax.text(
                    category_positions[subtype],
                    y_max + padding * 0.5,
                    marker,
                    ha="center",
                    color="red",
                    fontsize=14,
                )

    y_max = plot_data["Count"].max()
    y_min = plot_data["Count"].min()
    padding = max((y_max - y_min) * 0.1, 0.1 * max(abs(y_max), 1))
    ax.set_ylim(y_min - padding, y_max + padding)

    plt.title(f"Boxplot of {gene} by Molecular Subtype")
    plt.suptitle("")
    ax.set_xlabel("Molecular subtype")
    ax.set_ylabel("Normalized Gene Count")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()

    # Save plot to a string to display in HTML
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    ####Second plot: Pathologic diagnosis
    diagnosis_categories = [
        "Stroma",
        "NFT",
        "p53 sig",
        "STIL",
        "STIC",
        "HGSC",
    ]

    diagnosis_data = plot_data.copy()
    diagnosis_data["Diagnosis"] = diagnosis_data["Diagnosis"].astype("object")
    diagnosis_data.loc[diagnosis_data["Molecular_Subtype"] == "Stroma", "Diagnosis"] = (
        "Stroma"
    )
    diagnosis_data["Diagnosis_Grouped"] = diagnosis_data["Diagnosis"].replace(
        {"STICL": "STIC"}
    )
    diagnosis_data["Diagnosis_Grouped"] = pd.Categorical(
        diagnosis_data["Diagnosis_Grouped"],
        categories=diagnosis_categories,
        ordered=True,
    )
    diagnosis_data = diagnosis_data.dropna(subset=["Diagnosis_Grouped"])

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    diagnosis_data.boxplot(
        column="Count",
        by="Diagnosis_Grouped",
        ax=ax2,
        flierprops={"marker": ""},
    )

    diag_present = [
        cat
        for cat in diagnosis_categories
        if (diagnosis_data["Diagnosis_Grouped"] == cat).any()
    ]
    diag_positions = {cat: idx + 1 for idx, cat in enumerate(diag_present)}

    for subtype in diag_present:
        mask = diagnosis_data["Diagnosis_Grouped"] == subtype
        x = np.random.normal(diag_positions[subtype], jitter_strength, size=mask.sum())
        y = diagnosis_data.loc[mask, "Count"]
        ax2.scatter(x, y, color="black", alpha=0.7, s=point_size)

    nft_diag_counts = diagnosis_data.loc[
        diagnosis_data["Diagnosis_Grouped"] == "NFT", "Count"
    ]
    if len(nft_diag_counts) >= 2:
        for subtype in diag_present:
            if subtype in {"NFT", "Stroma"}:
                continue
            subtype_counts = diagnosis_data.loc[
                diagnosis_data["Diagnosis_Grouped"] == subtype, "Count"
            ]
            if len(subtype_counts) < 2:
                continue
            _, p_value = ttest_ind(nft_diag_counts, subtype_counts, equal_var=False)
            marker = None
            if p_value < 0.001:
                marker = "***"
            elif p_value < 0.01:
                marker = "**"
            elif p_value < 0.05:
                marker = "*"
            if marker:
                y_max = diagnosis_data["Count"].max()
                y_min = diagnosis_data["Count"].min()
                padding = max((y_max - y_min) * 0.1, 0.1 * max(abs(y_max), 1))
                ax2.text(
                    diag_positions[subtype],
                    y_max + padding * 0.5,
                    marker,
                    ha="center",
                    color="red",
                    fontsize=14,
                )

    diag_y_max = diagnosis_data["Count"].max()
    diag_y_min = diagnosis_data["Count"].min()
    diag_padding = max((diag_y_max - diag_y_min) * 0.1, 0.1 * max(abs(diag_y_max), 1))
    ax2.set_ylim(diag_y_min - diag_padding, diag_y_max + diag_padding)

    ax2.set_title("Boxplot of Pathologic Diagnosis")
    plt.suptitle("")
    ax2.set_xlabel("Pathologic diagnosis")
    ax2.set_ylabel("Normalized Gene Count")
    ax2.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.tick_params(axis="x", labelrotation=45)
    for tick in ax2.get_xticklabels():
        tick.set_ha("right")
    fig2.tight_layout()

    img2 = io.BytesIO()
    plt.savefig(img2, format="png")
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()
    plt.close(fig2)

    plot_url3 = None
    if selection and selection.lower() != "diagnosis":
        selection_data = pd.merge(
            gene_data,
            selected_annotations,
            left_on="Sample",
            right_on="SegmentDisplayName",
        ).dropna(subset=[selection])
        if not selection_data.empty:
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            selection_data.boxplot(
                column="Count",
                by=selection,
                ax=ax3,
                flierprops={"marker": ""},
            )

            cats = [lbl.get_text() for lbl in ax3.get_xticklabels()]
            for idx, cat in enumerate(cats, start=1):
                mask = selection_data[selection] == cat
                x = np.random.normal(idx, jitter_strength, size=mask.sum())
                y = selection_data.loc[mask, "Count"]
                ax3.scatter(x, y, color="black", alpha=0.7, s=point_size)

            ax3.set_title(f"Boxplot of {gene} by {selection.title()}")
            plt.suptitle("")
            ax3.set_xlabel(selection)
            ax3.set_ylabel("Normalized Gene Count")
            ax3.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax3.tick_params(axis="x", labelrotation=45)
            for tick in ax3.get_xticklabels():
                tick.set_ha("right")
            fig3.tight_layout()

            img3 = io.BytesIO()
            plt.savefig(img3, format="png")
            img3.seek(0)
            plot_url3 = base64.b64encode(img3.getvalue()).decode()
            plt.close(fig3)

    response = {"plot_url1": f"data:image/png;base64,{plot_url1}"}
    response["plot_url2"] = f"data:image/png;base64,{plot_url2}"
    response["plot_url3"] = (
        f"data:image/png;base64,{plot_url3}" if plot_url3 is not None else None
    )
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
