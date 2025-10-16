import base64
import io
import json
import logging
import os
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from scipy.stats import ttest_ind

matplotlib.use("Agg")

app = Flask(__name__)
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


def _warm_matplotlib():
    fig, ax = plt.subplots()
    ax.plot([0], [0])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)


def _warm_resources():
    try:
        if COUNT_FILE_ID:
            _ensure_local(COUNT_PATH, COUNT_FILE_ID)
        if ANNO_FILE_ID:
            _ensure_local(ANNO_PATH, ANNO_FILE_ID)
    except Exception as exc:
        logger.warning("Unable to prefetch data: %s", exc)
    try:
        _warm_matplotlib()
    except Exception as exc:
        logger.warning("Unable to warm matplotlib cache: %s", exc)


threading.Thread(target=_warm_resources, name="warm-resources", daemon=True).start()


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


@app.route("/")
def home():
    return render_template("pre_cancer_atlas.html")


@app.route("/pre_cancer_atlas", methods=["POST"])
def plot():
    gene = request.form["gene"].strip()

    # Load the count data and set index to "Gene"
    try:
        _ensure_local(COUNT_PATH, COUNT_FILE_ID)
        count_data = pd.read_parquet(COUNT_PATH).set_index("Gene")
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
    if gene not in count_data.index:
        return jsonify(error=f"The gene '{gene}' is not in the database."), 404

    # Select data for the chosen gene across all samples
    gene_data = count_data.loc[gene].to_frame(name="Count")
    gene_data.reset_index(inplace=True)
    gene_data.columns = ["Sample", "Count"]

    # Load annotation data and apply renaming and filtering
    try:
        _ensure_local(ANNO_PATH, ANNO_FILE_ID)
        anno_data = pd.read_parquet(
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
    rename_dict = {
        "Normal fallopian tube epithelium": "NFT",
        "p53 signature": "p53 sig",
        "STIC (incidental)": "STIC",
        "STIC-like lesion": "STICL",
        "High grade serous carcinoma": "HGSC",
    }
    anno_data["Diagnosis"] = anno_data["Diagnosis_ralph"].replace(rename_dict)
    relevant_diagnoses = ["NFT", "p53 sig", "STIL", "STIC", "STICL", "HGSC"]
    anno_data = anno_data[anno_data["Diagnosis"].isin(relevant_diagnoses)]
    anno_data["Diagnosis"] = pd.Categorical(
        anno_data["Diagnosis"], categories=relevant_diagnoses, ordered=True
    )

    # Select type
    sel_type = request.form.get("type", None)
    if sel_type:
        anno_data = anno_data[anno_data["type"] == sel_type]

    # Merge with gene data on 'Sample'
    merged_data = pd.merge(
        gene_data, anno_data, left_on="Sample", right_on="SegmentDisplayName"
    )
    # Ensure Molecular_Subtype has the correct categorical ordering
    merged_data["Molecular_Subtype"] = pd.Categorical(
        merged_data["Molecular_Subtype"],
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
    # Generate the boxplot with individual points
    fig, ax = plt.subplots()
    boxplot = merged_data.boxplot(
        column="Count",
        by="Molecular_Subtype",
        ax=ax,
        flierprops={"marker": ""},  # Hide default outliers
    )
    diagnoses = merged_data["Molecular_Subtype"].unique()

    # Overlay individual data points with reduced jitter
    jitter_strength = 0.06
    point_size = 5
    # iterate over the ordered Molecular_Subtype categories
    for i, subtype in enumerate(merged_data["Molecular_Subtype"].cat.categories):
        mask = merged_data["Molecular_Subtype"] == subtype
        x = np.random.normal(i + 1, jitter_strength, size=mask.sum())
        y = merged_data.loc[mask, "Count"]
        ax.scatter(x, y, color="black", alpha=0.7, s=point_size)

    # Statistical comparison against "NFT" on Molecular_Subtype
    nft_counts = merged_data.loc[merged_data["Molecular_Subtype"] == "NFT", "Count"]
    # iterate over the ordered subtypes, skipping the first ("NFT")
    for position, subtype in enumerate(
        merged_data["Molecular_Subtype"].cat.categories[1:], start=2
    ):
        subtype_counts = merged_data.loc[
            merged_data["Molecular_Subtype"] == subtype, "Count"
        ]
        t_stat, p_value = ttest_ind(nft_counts, subtype_counts)

        # choose marker based on p-value
        marker = None
        if p_value < 0.001:
            marker = "***"
        elif p_value < 0.01:
            marker = "**"
        elif p_value < 0.05:
            marker = "*"

        if marker:
            y_max = merged_data["Count"].max() * 1.05
            ax.text(position, y_max, marker, ha="center", color="red", fontsize=14)

    # Set y-axis limit slightly higher to fit asterisks within plot
    ax.set_ylim(merged_data["Count"].min(), merged_data["Count"].max() * 1.2)

    # Customize plot appearance
    plt.title(f"Boxplot of {gene} by Molecular Subtype")
    plt.suptitle("")  # Removes the automatic title by Pandas
    plt.xlabel("Molecular subtype")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Gene Count")
    plt.tight_layout()
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Save plot to a string to display in HTML
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    ####Second plot
    selection = request.form.get("selection", None)
    plot_url2 = None
    if selection and sel_type != "stroma":
        # Merge and drop missing
        plot_data = pd.merge(
            gene_data, anno_data, left_on="Sample", right_on="SegmentDisplayName"
        ).dropna(subset=[selection])

        # Draw boxplot
        fig2, ax2 = plt.subplots()
        plot_data.boxplot(
            column="Count", by=selection, ax=ax2, flierprops={"marker": ""}
        )

        # Jitter points
        jitter = 0.06
        size = 5
        cats = [lbl.get_text() for lbl in ax2.get_xticklabels()]
        for i, cat in enumerate(cats, start=1):
            mask = plot_data[selection] == cat
            x = np.random.normal(i, jitter, mask.sum())
            y = plot_data.loc[mask, "Count"]
            ax2.scatter(x, y, color="black", alpha=0.7, s=size)

        # Style
        ax2.set_title(f"Boxplot of {gene} by {selection.title()}")
        ax2.set_xlabel(selection)
        ax2.set_ylabel("Normalized Gene Count")
        ax2.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.suptitle("")  # remove auto “by …” label

        # Save it
        img2 = io.BytesIO()
        plt.savefig(img2, format="png")
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        plt.close(fig2)
    response = {"plot_url1": f"data:image/png;base64,{plot_url1}"}
    response["plot_url2"] = (
        f"data:image/png;base64,{plot_url2}" if plot_url2 is not None else None
    )
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
