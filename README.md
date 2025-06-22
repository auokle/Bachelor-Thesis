# Bachelor Thesis
This bachelor thesis is about the clustering of WHOOP data and finding the correlations between the clusters and survey data.

## File Structure

Short overview of the file structure:

- clustering_plots
- descriptive_statistics
  - descriptive_statistics_plots
- original_data
- working_data
  - cluster_labels

In original_data the following files have to be inserted before running the pipeline:
- mhs_demographics_sorted.csv
- mhs_sleep_sorted.csv
- mhs_survey_sorted.csv

## Pipeline

Multiple files have a section with parameters on top of the file, these parameters are used to load and store the file under the appropriate filenames. The pipeline is tested for the UMAP dimensionality reduction and three clusters across all survey categories. This should only be seen as an overview, the full explanation can be found in the thesis.

### 0. Preprocessing and Demographic Analysis

Before running the pipeline some preliminary files have to be created.

Running working_data/data_processing_working_data.ipynb will create the following files:
- working_data/mhs_sleep_ch.csv: The file with only the Swiss subpopulation of the whole mhs_sleep_sorted.csv file used throughout the pipeline.
- working_data/mhs_survey_sorted_without_nan.csv: The file with all the survey data but without entries that have NaN values used throughout the pipeline.
- working_data/sleep_intervals.csv: The file with sleep intervals used in the preliminary demographic analysis.
- working_data/sleep_userid_first_last_day.csv: The file with dates of the first and last time a user recorded sleep used in the preliminary demographic analysis.

Then the preliminary demographic analysis can be done by running all the .ipynb files in the descriptive_statistics folder.

### 1. Outlier Detection

Running data_preprocessing_outlier_detection_v3.ipynb can be used to remove outliers visually and save the outlier-free file under working_data/mhs_sleep_ch_without_outliers.csv.

### 2. Create Weekly Data

Running data_preprocessing_create_weekly.ipynb will create weekly datapoints from daily datapoints by aggregating them to a week and removing weeks with incomplete data. The resulting file is working_data/mhs_sleep_weekly.csv.

### 3. Feature Generation

Running data_preprocessing_create_features_v2.ipynb will create multiple features that capture the weekly data and store them under working_data/mhs_sleep_weekly_features.csv.

### 4. Feature Selection

Running data_preprocessing_feature_selection_v5.ipynb will chose features based on a maximal threshold of correlation between features, a minimal variance and by a minimal threshold with each survey category total score. The resulting files are stored under:
working_data/mhs_sleep_weekly_uncorr_features_correlation_threshold_{threshold}_{group_name}_{survey_category}.csv
Where {threshold} is the threshold of the correlation between the features, {group_name} is the group of the gender, and {survey_category} is the corresponding survey category.

### 5. Dimensionality Reduction

The file standardize_and_dim_red.py is called by main_caller_standardization_and_dim_red.ipynb, where it is possible to loop over several parameter combinations. It supports UMAP, PCA and TSNE, where only UMAP is fully tested in later pipeline stages. All files will be stored under working_data.

### 6. Clustering

Two clustering methods are supported: GMM and HAC. The files clustering_gmm_crossvalidation_v2.ipynb and clustering_HAC_crossvalidation.ipynb can be used to find the most suitable number of clusters by crossvalidation, where the plots have to be analysed manually and the number of clusters be defined for evaluation on a test set. This has to be done for each survey category individually and the corresponding labels of the whole dataset will be stored under working_data/cluster_labels.

### 7. Internal and External Validation

The file external_validation_GMM_HAC.ipynb is used to generate the clustering plots on the full data for both methods, and comparing the methods visually by an overlay and via evaluation methods. Before comparing clusters should be aligned as close as possible by changing the cluster names.

The file internal_validation_correlation_feature_cluster.ipynb will correlate the original selected features with the numberic values for the clusters by iterating over all permutations of the cluster assignments. For comparability the found cluster renamings of the external validation should be used in here. This has to be done for each clustering method individually.

### 8. Survey-Cluster Evaluation

The file correlation_feature_survey.ipynb will generate the demographic and survey total score distribution over the clusters. Then the correlation between the total scores and features, both adjusted for age, gender and BMI, will be generated.
