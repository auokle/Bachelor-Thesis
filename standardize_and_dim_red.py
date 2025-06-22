import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA
from pathlib import Path

def standardize(df, standardization):
    if (standardization):
        scaler = StandardScaler()
        numerical_cols = df.columns[2:] # skip USER_ID nad WEEK_START
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
def dim_red(df, method, correlation_threshold, perplexity, standardization, group, survey_category, force_recreate=False):
    if method == "tsne":
        filename = f"working_data/{method}_features_correlation_threshold_{correlation_threshold}_perplexity_{perplexity}_{'with' if standardization else 'without'}_standardization_{group}_{survey_category}.csv"
    else:
        filename = f"working_data/{method}_features_correlation_threshold_{correlation_threshold}_{'with' if standardization else 'without'}_standardization_{group}_{survey_category}.csv"

    file_path = Path(filename)

    if file_path.exists() and not force_recreate:
        print(f"Loading existing dimensionality reduction from {filename}")
        return pd.read_csv(file_path)

    id_cols = df.iloc[:, :2].copy()
    feature_cols = df.iloc[:, 2:]

    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=1, n_jobs=1)
        reduced_data = tsne.fit_transform(feature_cols)
        reduced_df = pd.DataFrame(reduced_data, columns=["TSNE_1", "TSNE_2"])
    elif method == "umap":
        umap_model = umap.UMAP(n_components=2, random_state=1)
        reduced_data = umap_model.fit_transform(feature_cols)
        reduced_df = pd.DataFrame(reduced_data, columns=["UMAP_1", "UMAP_2"])
    elif method == "pca":
        pca = PCA(n_components=2, random_state=1)
        reduced_data = pca.fit_transform(feature_cols)
        reduced_df = pd.DataFrame(reduced_data, columns=["PCA_1", "PCA_2"])
    else:
        raise Exception("Chosen method is not defined.")

    reduced_df.index = id_cols.index
    df = pd.concat([id_cols, reduced_df], axis=1)

    df.to_csv(filename, index=False)
    print(f"Saved reduced data to {filename}")
    return df
            
def all_plots(df, method, correlation_threshold, perplexity, standardization, group, survey_category, force_recreate=False):
    
    def build_filename(plot_type):
        prefix = f"{method.upper()}_plot_{plot_type}_correlation_threshold_{correlation_threshold}"
        if method == "tsne":
            prefix += f"_and_perplexity_{perplexity}"
        prefix += f"_{'with' if standardization else 'without'}_standardization_{group}_{survey_category}"
        return f"clustering_plots/{prefix}.png"

    # Scatter Plot
    scatter_file = Path(build_filename("scatter"))
    if not scatter_file.exists() or force_recreate:
        plt.figure()
        plt.scatter(df[f"{method.upper()}_1"], df[f"{method.upper()}_2"], s=1, alpha=0.2)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        title = f"{method.upper()} scatter plot with correlation threshold {correlation_threshold}"
        if method == "tsne":
            title += f" and perplexity {perplexity}"
        title += " with standardization" if standardization else " without standardization"
        #plt.title(title + f" ({group})")
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        plt.show()
    else:
        print(f"Scatter plot already exists: {scatter_file}")

    # Hexbin Plot
    hexbin_file = Path(build_filename("hexbin"))
    if not hexbin_file.exists() or force_recreate:
        plt.figure()
        plt.hexbin(df[f"{method.upper()}_1"], df[f"{method.upper()}_2"], gridsize=50, cmap="coolwarm")
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.colorbar(label="Density")
        title = f"{method.upper()} Density Plot (Hexbin) with correlation threshold {correlation_threshold}"
        if method == "tsne":
            title += f" and perplexity {perplexity}"
        title += " with standardization" if standardization else " without standardization"
        #plt.title(title + f" ({group})")
        plt.savefig(hexbin_file, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        plt.show()
    else:
        print(f"Hexbin plot already exists: {hexbin_file}")

    # KDE Contour Plot
    kde_file = Path(build_filename("kde_contour"))
    if not kde_file.exists() or force_recreate:
        plt.figure()
        sns.kdeplot(x=df[f"{method.upper()}_1"], y=df[f"{method.upper()}_2"], cmap="coolwarm", levels=20)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        title = f"{method.upper()} contour density plot with correlation threshold {correlation_threshold}"
        if method == "tsne":
            title += f" and perplexity {perplexity}"
        title += " with standardization" if standardization else " without standardization"
        #plt.title(title + f" ({group})")
        plt.savefig(kde_file, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
        plt.show()
    else:
        print(f"KDE contour plot already exists: {kde_file}")
    
        
def standardize_and_dim_red(method, correlation_threshold, perplexity, standardization, group, survey_category, force_recreate=False):
    input_filename = f"working_data/mhs_sleep_weekly_uncorr_features_correlation_threshold_{correlation_threshold}_{group}_{survey_category}.csv"

    try:
        df = pd.read_csv(input_filename)
        print(f"Loaded data from {input_filename}")
    except FileNotFoundError:
        raise Exception(f"Input file not found: {input_filename}")

    standardize(df, standardization)
    df = dim_red(df, method, correlation_threshold, perplexity, standardization, group, survey_category, force_recreate)
    all_plots(df, method, correlation_threshold, perplexity, standardization, group, survey_category, force_recreate)
    