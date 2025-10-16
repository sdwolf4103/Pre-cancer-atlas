from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from scipy.stats import ttest_ind

import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("pre_cancer_atlas.html")


@app.route("/pre_cancer_atlas", methods=["POST"])
def plot():
    gene = request.form["gene"].strip()

    # Load the count data and set index to "Gene"
    count_data = pd.read_parquet(
        "/Users/sd_wo/Library/CloudStorage/OneDrive-JohnsHopkins/projects/nanostring/share/Nanostring/Data/count_data.parquet"
    ).set_index("Gene")

    # Check if the gene exists
    if gene not in count_data.index:
        return jsonify(error=f"The gene '{gene}' is not in the database."), 404

    # Select data for the chosen gene across all samples
    gene_data = count_data.loc[gene].to_frame(name="Count")
    gene_data.reset_index(inplace=True)
    gene_data.columns = ["Sample", "Count"]

    # Load annotation data and apply renaming and filtering
    anno_data = pd.read_parquet(
        "/Users/sd_wo/Library/CloudStorage/OneDrive-JohnsHopkins/projects/nanostring/share/Nanostring/Data/anno_data.parquet",
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
    return jsonify(
        plot_url1=f"data:image/png;base64,{plot_url1}",
        plot_url2=f"data:image/png;base64,{plot_url2}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
