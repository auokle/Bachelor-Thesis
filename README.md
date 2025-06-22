# Bachelor Thesis

This repository contains the code and data processing pipeline for a bachelor thesis focused on clustering WHOOP-derived physiological data and examining correlations between resulting clusters and self-reported mental health survey data.

## File Structure

A brief overview of the directory structure:

- `clustering_plots/`
- `descriptive_statistics/`
  - `descriptive_statistics_plots/`
- `original_data/`
- `working_data/`
  - `cluster_labels/`

Before running the pipeline, insert the following original data files into `original_data/`:

- `mhs_demographics_sorted.csv`
- `mhs_sleep_sorted.csv`
- `mhs_survey_sorted.csv`

## Pipeline Overview

Many files include a parameter section at the top to define filenames and output paths. The pipeline was developed for UMAP-based dimensionality reduction and three-cluster solutions across all survey categories. This README provides a high-level overview — for detailed methodology, please refer to the thesis document.

### 0. Preprocessing and Demographic Analysis

Run `working_data/data_processing_working_data.ipynb` to generate:

- `mhs_sleep_ch.csv`: Subset of Swiss users.
- `mhs_survey_sorted_without_nan.csv`: Survey data excluding entries with NaN values.
- `sleep_intervals.csv`: Sleep intervals for demographic analysis.
- `sleep_userid_first_last_day.csv`: First/last sleep records per user.

Demographic analyses can be performed by executing the `.ipynb` notebooks in the `descriptive_statistics/` folder.

### 1. Outlier Detection

Run `data_preprocessing_outlier_detection_v3.ipynb` to detect and visually remove outliers. The cleaned output is saved as:

- `working_data/mhs_sleep_ch_without_outliers.csv`

### 2. Weekly Aggregation

Run `data_preprocessing_create_weekly.ipynb` to convert daily data into weekly summaries. Incomplete weeks are removed. Output:

- `working_data/mhs_sleep_weekly.csv`

### 3. Feature Generation

Run `data_preprocessing_create_features_v2.ipynb` to compute derived weekly features from the aggregated data. Output:

- `working_data/mhs_sleep_weekly_features.csv`

### 4. Feature Selection

Run `data_preprocessing_feature_selection_v5.ipynb` to select features based on:
- Maximum inter-feature correlation threshold,
- Minimum variance,
- Minimum correlation with survey total score.

Resulting files are named:
`mhs_sleep_weekly_uncorr_features_correlation_threshold_{threshold}_{group_name}_{survey_category}.csv`


### 5. Dimensionality Reduction

Run `main_caller_standardization_and_dim_red.ipynb`, which calls `standardize_and_dim_red.py`. Supports UMAP, PCA, and t-SNE (only UMAP fully tested in later stages). Outputs are saved to `working_data/`.

### 6. Clustering

Clustering is supported via:
- Gaussian Mixture Models (GMM): `clustering_gmm_crossvalidation_v2.ipynb`
- Hierarchical Agglomerative Clustering (HAC): `clustering_HAC_crossvalidation.ipynb`

Cross-validation plots are generated for manual selection of the optimal number of clusters (done separately for each survey category). Final cluster labels are saved in `working_data/cluster_labels/`.

### 7. Internal and External Validation

Run `external_validation_GMM_HAC.ipynb` to:
- Generate cluster visualizations (GMM vs. HAC),
- Visually and quantitatively compare clustering results.

Before comparison, manually align clusters between methods by renaming for maximal overlap.

Run `internal_validation_correlation_feature_cluster.ipynb` to:
- Correlate features with cluster labels,
- Iterate over all cluster label permutations to maximize correlation with survey scores.

This should be run separately for each method (GMM and HAC) using the cluster label mappings established in external validation.

### 8. Survey-Cluster Evaluation

Run `correlation_feature_survey.ipynb` to:
- Visualize demographic and survey score distributions across clusters,
- Compute partial correlations (adjusted for age, gender, BMI) between features and survey total scores.
