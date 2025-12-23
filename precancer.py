import base64
import io
import json
import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    import seaborn as sns
except ImportError:  # seaborn optional
    sns = None
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from scipy.stats import ttest_ind, pearsonr
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

try:
    import google.cloud.logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    # Initialize logging client - lazy load to avoid errors if no creds
    logging_client = google.cloud.logging.Client()
    logging_handler = logging_client.get_default_handler()
    cloud_logger = logging.getLogger("usage_logger")
    cloud_logger.setLevel(logging.INFO)
    cloud_logger.addHandler(logging_handler)
except Exception:
    cloud_logger = None

def log_usage_event(event_type: str, details: dict):
    """Log a structured usage event to Cloud Logging."""
    # Capture visitor context
    try:
        # get_remote_address handles X-Forwarded-For if configured properly, 
        # or we can use request.headers.get("X-Forwarded-For") directly for Cloud Run.
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ua = request.user_agent.string
    except Exception:
        ip = "unknown"
        ua = "unknown"

    payload = {
        "event_type": event_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "ip_address": ip,
        "user_agent": ua,
        **details
    }
    if cloud_logger:
        cloud_logger.info(json.dumps(payload))
    else:
        # Fallback to stdout for local dev
        print(f"USAGE_LOG: {json.dumps(payload)}")

HERE = Path(__file__).resolve().parent


def _resolve_asset_dir(name: str) -> Path:
    """Return the first matching asset directory walking up the tree."""
    for candidate in (HERE / name, HERE.parent / name, HERE.parent.parent / name):
        if candidate.exists():
            return candidate
    return HERE / name


STATIC_DIR = _resolve_asset_dir("static")
TEMPLATE_DIR = _resolve_asset_dir("templates")

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
    template_folder=str(TEMPLATE_DIR),
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="You have exceeded the request limit. Please try again later."), 429

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

CATEGORY_COLUMN_MAP = {
    "molecular": "Molecular_Subtype",
    "morphology": "Morphology category",
    "brca": "BRCA category",
    "diagnosis": "Diagnosis",
}

CATEGORY_LABEL_MAP = {
    "Molecular_Subtype": "Molecular category",
    "Morphology category": "Morphology category",
    "BRCA category": "BRCA category",
    "Diagnosis": "Diagnosis",
}

DEFAULT_CORRELATION_TYPE = "epithelium"
MOLECULAR_SUBTYPE_ORDER = [
    "Stroma",
    "NFT",
    "Dormant",
    "Mixed",
    "Immunoreactive",
    "Proliferative",
    "HGSC",
]

_COUNT_PARQUET: Optional[pq.ParquetFile] = None
_COUNT_PARQUET_LOCK = threading.Lock()
_GENE_ROW_GROUP_CACHE: Dict[str, List[int]] = {}
_GENE_ROW_GROUP_CACHE_LOCK = threading.Lock()
_ANNOTATION_CACHE: Dict[Optional[str], pd.DataFrame] = {}
_ANNOTATION_CACHE_LOCK = threading.Lock()


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


def _get_count_parquet() -> pq.ParquetFile:
    global _COUNT_PARQUET
    with _COUNT_PARQUET_LOCK:
        if _COUNT_PARQUET is None:
            _ensure_local(COUNT_PATH, COUNT_FILE_ID)
            _COUNT_PARQUET = pq.ParquetFile(COUNT_PATH)
    return _COUNT_PARQUET


def _get_gene_row_groups(gene: str, pf: pq.ParquetFile) -> List[int]:
    normalized = (gene or "").strip().upper()
    if not normalized:
        return []

    with _GENE_ROW_GROUP_CACHE_LOCK:
        cached = _GENE_ROW_GROUP_CACHE.get(normalized)
    if cached is not None:
        return cached

    gene_scalar = pa.scalar(normalized)
    matches: List[int] = []
    for rg in range(pf.num_row_groups):
        gene_only = pf.read_row_group(rg, columns=["Gene"])
        mask = pc.equal(gene_only.column("Gene"), gene_scalar)
        if pc.any(mask).as_py():
            matches.append(rg)

    with _GENE_ROW_GROUP_CACHE_LOCK:
        _GENE_ROW_GROUP_CACHE[normalized] = matches
    return matches


def _read_gene_row(gene: str) -> pd.DataFrame:
    normalized = (gene or "").strip().upper()
    if not normalized:
        return pd.DataFrame()

    pf = _get_count_parquet()
    row_groups = _get_gene_row_groups(normalized, pf)
    if not row_groups:
        return pd.DataFrame()

    gene_scalar = pa.scalar(normalized)
    tables = []
    for rg in row_groups:
        full = pf.read_row_group(rg)
        mask = pc.equal(full.column("Gene"), gene_scalar)
        if not pc.any(mask).as_py():
            continue
        tables.append(full.filter(mask))

    if not tables:
        with _GENE_ROW_GROUP_CACHE_LOCK:
            _GENE_ROW_GROUP_CACHE[normalized] = []
        return pd.DataFrame()

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return table.to_pandas()


def _extract_gene_counts(gene: str) -> pd.DataFrame:
    """Return count values for a single gene indexed by sample."""
    if not gene:
        return pd.DataFrame()

    try:
        gene_frame = _read_gene_row(gene)
    except Exception as exc:
        logger.warning("Unable to read gene row for %s: %s", gene, exc)
        raise

    if gene_frame.empty:
        return pd.DataFrame()

    gene_frame = gene_frame.set_index("Gene")
    try:
        series = gene_frame.loc[gene]
    except KeyError:
        return pd.DataFrame()

    counts = series.to_frame(name="Count").reset_index()
    counts.columns = ["Sample", "Count"]
    counts["Sample"] = counts["Sample"].astype(str)
    return counts





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
    log_usage_event("VISIT", {})
    return render_template("pre_cancer_atlas.html")


@app.route("/robots.txt")
def robots_txt():
    return send_from_directory(STATIC_DIR, "robots.txt", mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    return send_from_directory(STATIC_DIR, "sitemap.xml", mimetype="application/xml")





@app.route("/pre_cancer_atlas", methods=["POST"])
@limiter.limit("20 per minute")
def plot():
    gene = request.form["gene"].strip().upper()
    
    log_usage_event("PLOT_GENERATED", {"gene": gene})

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

    categories_order = MOLECULAR_SUBTYPE_ORDER

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

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data.boxplot(
        column="Count",
        by="Molecular_Subtype",
        ax=ax,
        flierprops={"marker": ""},
        grid=False,
    )

    jitter_strength = 0.06
    point_size = 5
    for subtype in present_categories:
        mask = plot_data["Molecular_Subtype"] == subtype
        x = np.random.normal(
            category_positions[subtype], jitter_strength, size=mask.sum()
        )
        y = plot_data.loc[mask, "Count"]
        ax.scatter(x, y, color="black", alpha=0.6, s=point_size)

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

    ax.set_title(f"{gene} Expression by Molecular Subtypes", fontsize=14)
    fig.suptitle("")
    ax.set_xlabel("Molecular subtype", fontsize=12)
    ax.set_ylabel("Normalized Gene Count", fontsize=12)
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="both", labelsize=10)
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

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    diagnosis_data.boxplot(
        column="Count",
        by="Diagnosis_Grouped",
        ax=ax2,
        flierprops={"marker": ""},
        grid=False,
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
        ax2.scatter(x, y, color="black", alpha=0.6, s=point_size)

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

    ax2.set_title(f"{gene} Expression by Pathologic Diagnosis", fontsize=14)
    fig2.suptitle("")
    ax2.set_xlabel("Pathologic diagnosis", fontsize=12)
    ax2.set_ylabel("Normalized Gene Count", fontsize=12)
    ax2.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.tick_params(axis="both", labelsize=10)
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
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            selection_data.boxplot(
                column="Count",
                by=selection,
                ax=ax3,
                flierprops={"marker": ""},
                grid=False,
            )

            cats = [lbl.get_text() for lbl in ax3.get_xticklabels()]
            for idx, cat in enumerate(cats, start=1):
                mask = selection_data[selection] == cat
                x = np.random.normal(idx, jitter_strength, size=mask.sum())
                y = selection_data.loc[mask, "Count"]
                ax3.scatter(x, y, color="black", alpha=0.6, s=point_size)

            ax3.set_title(f"{gene} Expression by {selection.title()}", fontsize=14)
            fig3.suptitle("")
            ax3.set_xlabel(selection, fontsize=12)
            ax3.set_ylabel("Normalized Gene Count", fontsize=12)
            ax3.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax3.tick_params(axis="both", labelsize=10)
            ax3.tick_params(axis="x", labelrotation=45)
            for tick in ax3.get_xticklabels():
                tick.set_ha("right")
            fig3.tight_layout()

            img3 = io.BytesIO()
            plt.savefig(img3, format="png")
            img3.seek(0)
            plot_url3 = base64.b64encode(img3.getvalue()).decode()
            plt.close(fig3)

    response = {"plot_url1": f"data:image/png;base64,{plot_url2}"}
    response["plot_url2"] = f"data:image/png;base64,{plot_url1}"
    response["plot_url3"] = (
        f"data:image/png;base64,{plot_url3}" if plot_url3 is not None else None
    )
    return jsonify(response)


@app.route("/pre_cancer_correlation", methods=["POST"])
@limiter.limit("20 per minute")
def correlation():
    gene_a = (request.form.get("geneA") or "").strip().upper()
    gene_b = (request.form.get("geneB") or "").strip().upper()
    
    log_usage_event("CORRELATION_GENERATED", {"gene_a": gene_a, "gene_b": gene_b})

    if not gene_a or not gene_b:
        return jsonify(error="Both Gene A and Gene B are required."), 400

    sel_type = (request.form.get("type") or DEFAULT_CORRELATION_TYPE).strip().lower()
    if sel_type not in {"epithelium", "stroma"}:
        sel_type = DEFAULT_CORRELATION_TYPE

    category_key = (request.form.get("category") or "").strip()
    detail_values = [
        value for value in request.form.getlist("category_detail") if value
    ]

    try:
        _ensure_local(COUNT_PATH, COUNT_FILE_ID)
        gene_a_counts = _extract_gene_counts(gene_a)
        gene_b_counts = _extract_gene_counts(gene_b)
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
        return jsonify(error=f"Unable to load gene counts: {exc}"), 500

    if gene_a_counts.empty:
        return jsonify(error=f"The gene '{gene_a}' is not in the database."), 404
    if gene_b_counts.empty:
        return jsonify(error=f"The gene '{gene_b}' is not in the database."), 404

    gene_a_counts = gene_a_counts.rename(columns={"Count": "gene_a_count"})
    gene_b_counts = gene_b_counts.rename(columns={"Count": "gene_b_count"})

    try:
        annotations = _prepare_annotation(sel_type)
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
        return jsonify(error=f"Unable to load annotation data: {exc}"), 500

    annotations = annotations.copy()
    annotations["Sample"] = annotations["SegmentDisplayName"].astype(str)

    category_column = CATEGORY_COLUMN_MAP.get(category_key)
    summary_detail_values = detail_values.copy()
    if category_column:
        annotations = annotations.dropna(subset=[category_column])
        if detail_values:
            filter_values: set = set()
            if "*" in detail_values:
                detail_values = []
                summary_detail_values = (
                    annotations[category_column].dropna().unique().tolist()
                )
            if detail_values:
                if category_column == "BRCA category":
                    unique_values = annotations[category_column].dropna().unique()
                    normalized_unique = [
                        (str(value).strip(), value) for value in unique_values
                    ]
                    for selected in detail_values:
                        if selected == "Negative":
                            for text, original in normalized_unique:
                                if text.lower() == "negative":
                                    filter_values.add(original)
                        elif selected == "Other":
                            for text, original in normalized_unique:
                                lower = text.lower()
                                if (
                                    text not in {"BRCA1", "BRCA2"}
                                    and lower != "negative"
                                ):
                                    filter_values.add(original)
                        else:
                            filter_values.add(selected)
                else:
                    filter_values = set(detail_values)
                    if category_column == "Diagnosis" and "STIC" in filter_values:
                        filter_values.add("STICL")
                if filter_values:
                    annotations = annotations[
                        annotations[category_column].isin(filter_values)
                    ]

    if annotations.empty:
        return (
            jsonify(error="No samples available after applying the selected filters."),
            404,
        )

    sample_frame = annotations[["Sample"]].drop_duplicates()
    merged = sample_frame.merge(gene_a_counts, on="Sample", how="inner").merge(
        gene_b_counts, on="Sample", how="inner"
    )
    merged = merged.dropna(subset=["gene_a_count", "gene_b_count"])

    if merged.empty:
        return jsonify(error="No overlapping samples for the selected genes."), 404

    merged = merged.drop_duplicates(subset=["Sample"])

    sample_count = len(merged)
    if sample_count < 3:
        return (
            jsonify(
                error="At least three overlapping samples are required for correlation."
            ),
            400,
        )

    x = merged["gene_a_count"].astype(float).to_numpy()
    y = merged["gene_b_count"].astype(float).to_numpy()

    try:
        r_value, p_value = pearsonr(x, y)
    except ValueError as exc:
        return jsonify(error=f"Unable to compute correlation: {exc}"), 400

    slope = intercept = None
    x_line = y_line = None
    if np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x,
        y,
        color="#1f3b70",
        edgecolors="#ffffff",
        linewidths=0.5,
        alpha=0.75,
        s=54,
    )
    if x_line is not None and y_line is not None:
        ax.plot(x_line, y_line, color="#8b1a1a", linewidth=1.6)

    ax.set_xlabel(f"{gene_a} normalized count", fontsize=12)
    ax.set_ylabel(f"{gene_b} normalized count", fontsize=12)
    ax.set_title(f"{gene_a} - {gene_b} correlation", fontsize=14)
    ax.grid(axis="both", color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    stats_text = f"Pearson r = {r_value:.3f}, p = {p_value:.3e} (n = {sample_count})"

    filter_parts: list[str] = [f"Type: {sel_type.title()}"]
    if category_column:
        category_label = CATEGORY_LABEL_MAP.get(category_column, category_column)
        display_values: list[str] = []
        seen_display: set[str] = set()
        for value in summary_detail_values:
            display_value = value
            if category_column == "Diagnosis" and value in {"STIC", "STICL"}:
                display_value = "STIC"
            if display_value and display_value not in seen_display:
                seen_display.add(display_value)
                display_values.append(display_value)

        if display_values:
            filter_parts.append(f"{category_label}: {', '.join(display_values)}")
        else:
            filter_parts.append(category_label)
    filter_summary = "; ".join(filter_parts)

    response = {
        "plot_url": f"data:image/png;base64,{plot_b64}",
        "stats_text": stats_text,
        "sample_count": sample_count,
        "summary": filter_summary,
    }
    return jsonify(response)


@app.route("/pre_cancer_heatmap", methods=["POST"])
def heatmap():
    heatmap_type = (request.form.get("heatmapType") or "stroma").strip().lower()
    if heatmap_type not in {"epithelium", "stroma", "both"}:
        heatmap_type = "stroma"

    gene_input = request.form.get("heatmapGenes") or ""
    tokens = [
        token.strip().upper()
        for token in re.split(r"[,;\s]+", gene_input)
        if token.strip()
    ]
    # Preserve order while removing duplicates
    seen: Dict[str, None] = {}
    genes: List[str] = []
    for token in tokens:
        if token not in seen:
            seen[token] = None
            genes.append(token)

    if not genes:
        return jsonify(error="Please enter at least one gene symbol."), 400

    max_genes = 25
    if len(genes) > max_genes:
        genes = genes[:max_genes]

    group_mode = (request.form.get("heatmapCategory") or "diagnosis").strip().lower()
    if group_mode not in {"molecular", "diagnosis"}:
        group_mode = "diagnosis"

    scale_mode = (request.form.get("heatmapScale") or "raw").strip().lower()
    if scale_mode not in {"raw", "zscore"}:
        scale_mode = "raw"
    scale_label = "Mean normalized count" if scale_mode == "raw" else "Per-gene z-score"

    try:
        annotation_sets: List[tuple[pd.DataFrame, Optional[str], bool]] = []
        if heatmap_type in {"epithelium", "both"}:
            annotation_sets.append((_prepare_annotation("epithelium"), None, False))
        if heatmap_type in {"stroma", "both"}:
            collapse_stroma = heatmap_type == "both"
            forced_label = "Stroma" if collapse_stroma else None
            annotation_sets.append(
                (_prepare_annotation("stroma"), forced_label, collapse_stroma)
            )
        if not annotation_sets:
            return jsonify(error="Unable to load annotation data."), 500
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
        return jsonify(error=f"Unable to load annotation data: {exc}"), 500

    if group_mode == "diagnosis":
        category_order = ["Stroma", "NFT", "p53 sig", "STIL", "STIC", "HGSC"]
    else:
        category_order = MOLECULAR_SUBTYPE_ORDER

    expression_matrix = pd.DataFrame(index=category_order, dtype=float)
    missing_genes: List[str] = []

    def _prepare_group_frame(
        frame: pd.DataFrame, forced_label: Optional[str], collapse_to_stroma: bool
    ) -> pd.DataFrame:
        df = frame.copy()
        if forced_label:
            df["Molecular_Subtype"] = forced_label
        df["Molecular_Subtype"] = pd.Categorical(
            df["Molecular_Subtype"], categories=MOLECULAR_SUBTYPE_ORDER, ordered=True
        )
        df = df.dropna(subset=["Molecular_Subtype"])
        if df.empty:
            return df
        if collapse_to_stroma:
            df["Heatmap_Group"] = "Stroma"
        elif group_mode == "diagnosis":
            df["Heatmap_Group"] = df["Diagnosis"].astype("object")
            if forced_label is None:
                df.loc[df["Molecular_Subtype"] == "Stroma", "Heatmap_Group"] = "Stroma"
            df["Heatmap_Group"] = df["Heatmap_Group"].replace({"STICL": "STIC"})
        else:
            df["Heatmap_Group"] = df["Molecular_Subtype"]
        df["Heatmap_Group"] = pd.Categorical(
            df["Heatmap_Group"], categories=category_order, ordered=True
        )
        df = df.dropna(subset=["Heatmap_Group"])
        return df

    group_sample_sets: Dict[str, set[str]] = {cat: set() for cat in category_order}
    for annotations, forced_label, collapse_to_stroma in annotation_sets:
        grouped = _prepare_group_frame(annotations, forced_label, collapse_to_stroma)
        if grouped.empty:
            continue
        for group_value, sample_id in grouped[
            ["Heatmap_Group", "SegmentDisplayName"]
        ].itertuples(index=False):
            if pd.isna(group_value):
                continue
            group_sample_sets.setdefault(group_value, set()).add(str(sample_id))

    group_stats: Dict[str, int] = {
        cat: len(samples) for cat, samples in group_sample_sets.items()
    }

    for gene in genes:
        try:
            counts = _extract_gene_counts(gene)
        except Exception as exc:
            logger.warning("Unable to extract counts for %s: %s", gene, exc)
            missing_genes.append(gene)
            continue

        if counts.empty:
            missing_genes.append(gene)
            continue

        merged_frames: List[pd.DataFrame] = []
        for annotations, forced_label, collapse_to_stroma in annotation_sets:
            subset = counts.merge(
                annotations, left_on="Sample", right_on="SegmentDisplayName"
            )
            if subset.empty:
                continue
            grouped = _prepare_group_frame(
                subset, forced_label, collapse_to_stroma
            )
            if grouped.empty:
                continue
            merged_frames.append(grouped[["Heatmap_Group", "Count"]])

        if not merged_frames:
            missing_genes.append(gene)
            continue

        combined = pd.concat(merged_frames, ignore_index=True)
        if scale_mode == "zscore":
            values = combined["Count"].astype(float)
            if not values.empty:
                mean_val = values.mean()
                std_val = values.std(ddof=0)
                if std_val > 0 and not np.isclose(std_val, 0):
                    combined["Count"] = (values - mean_val) / std_val
                else:
                    combined["Count"] = values - mean_val
        combined["Heatmap_Group"] = pd.Categorical(
            combined["Heatmap_Group"], categories=category_order, ordered=True
        )
        group_means = combined.groupby("Heatmap_Group", observed=True)["Count"]
        averages = group_means.mean()
        expression_matrix[gene] = averages

    expression_matrix = expression_matrix.dropna(how="all")
    if expression_matrix.empty:
        message = "No expression data found for the selected genes."
        if missing_genes:
            message += f" Missing genes: {', '.join(missing_genes)}."
        return jsonify(error=message.strip()), 404

    # Build heatmap (palette follows selected scaling mode)
    plt.close("all")
    num_categories = len(expression_matrix.index)  # lesion groups on x
    num_genes = len(expression_matrix.columns)  # genes on y
    cell_size = 0.6
    width_margin = 2.2  # allow space for labels + color bar
    height_margin = 2.0
    fig_width = max(5.0, num_categories * cell_size + width_margin)
    fig_height = max(4.0, num_genes * cell_size + height_margin)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    data_for_heatmap = expression_matrix.T
    cmap_name = "YlOrRd" if scale_mode == "raw" else "coolwarm"
    if sns is not None:
        sns.set_style("white")
        cmap = sns.color_palette(cmap_name, as_cmap=True)
        sns.heatmap(
            data_for_heatmap,
            ax=ax,
            cmap=cmap,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": scale_label},
            annot=True,
            fmt=".1f",
        )
    else:
        data = np.ma.masked_invalid(data_for_heatmap.to_numpy(dtype=float))
        try:
            cmap = plt.cm.get_cmap(cmap_name).copy()
        except ValueError:
            fallback = "coolwarm" if scale_mode == "zscore" else "magma_r"
            cmap = plt.cm.get_cmap(fallback).copy()
        cmap.set_bad(color="#f5f5f5")
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(data_for_heatmap.columns)))
        ax.set_xticklabels(
            data_for_heatmap.columns, rotation=45, ha="right", fontsize=10
        )
        ax.set_yticks(np.arange(len(data_for_heatmap.index)))
        ax.set_yticklabels(data_for_heatmap.index, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(scale_label, fontsize=10)
        valid_values = data_for_heatmap.to_numpy(dtype=float)
        finite_values = valid_values[np.isfinite(valid_values)]
        mean_value = finite_values.mean() if finite_values.size else 0
        for i, row_label in enumerate(data_for_heatmap.index):
            for j, gene_label in enumerate(data_for_heatmap.columns):
                value = data_for_heatmap.iloc[i, j]
                if pd.isna(value):
                    continue
                text_color = "#ffffff" if value >= mean_value else "#111111"
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    ax.set_xlabel("Lesion group", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)
    ax.set_title(scale_label, fontsize=12, pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=9, labelright=False, labeltop=False)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", pad_inches=0.15)
    buffer.seek(0)
    plot_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    response = {
        "heatmap_url": f"data:image/png;base64,{plot_b64}",
        "missing_genes": missing_genes,
        "category_count": len(expression_matrix.index),
        "scale_label": scale_label,
        "group_stats": [
            f"{label}: {group_stats.get(label, 0)} samples"
            for label in category_order
            if group_stats.get(label, 0)
        ],
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
