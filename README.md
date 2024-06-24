# Detecting-Presence-of-Huntington-Disease-Using-Machine-Learning-Models-on-Gene-Expression-Data

## Project Overview

This project aims to detect the presence of Huntington's Disease (HD) using machine learning models applied to gene expression data. HD is a neurodegenerative disease caused by a mutation in the HTT gene, leading to progressive deterioration of physical, mental, and emotional health. Early detection of HD can significantly improve the quality of life for patients. This project uses Support Vector Machine (SVM), Random Forest, and XGBoost models to classify gene expression data for detecting HD.

## Problem Statement

The primary goal is to classify gene expression data to detect the presence of Huntington's Disease using machine learning models. The dataset used consists of 157 HD positive samples and 157 control samples with 273 features. The project involves data preprocessing, dimensionality reduction using PCA, and training supervised machine learning models.

## Project Structure

### 1. Data Acquisition
- **Source**: [GEO Dataset GSE33000](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=gse33000)
- **Description**: The dataset contains microarray gene expression profiles of the prefrontal cortex brain tissues of HD patients and non-demented control samples.

### 2. Data Preprocessing
- Normalization using StandardScaler()
- Scaling using MinMaxScaler()
- Dimensionality reduction using PCA to select the top 10 principal components

### 3. Model Training and Evaluation
- **Algorithms Used**:
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - XGBoost
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - K-fold cross validation

## Scripts Description

### `data_preprocessing.ipynb`
This notebook handles the preprocessing of the GSE33000 dataset, including:
- Loading the dataset
- Normalizing and scaling the data
- Dimensionality reduction using PCA
  
This notebook also handles the training and evaluation of machine learning models, including:
- Splitting the dataset into training and testing sets
- Training SVM, Random Forest, and XGBoost models
- Evaluating model performance using accuracy, precision, recall, F1-score, confusion matrix, and K-fold cross validation

This notebook finally visualizes the results, including confusion matrices and performance metrics for each model.

## Results

- **Support Vector Machine (SVM)**:
  - Accuracy: 0.94
  - F1-score: 0.93
  - Confusion Matrix: 
    - True Negative: 32
    - False Positive: 3
    - False Negative: 1
    - True Positive: 27
  - K-fold cross validation results: Average test score of 0.94

- **Random Forest**:
  - Accuracy: 0.91
  - F1-score: 0.83
  - Confusion Matrix: 
    - True Negative: 30
    - False Positive: 3
    - False Negative: 5
    - True Positive: 25
  - K-fold cross validation results: Average test score of 0.91

- **XGBoost**:
  - Accuracy: 0.91
  - F1-score: 0.83
  - Confusion Matrix: 
    - True Negative: 30
    - False Positive: 5
    - False Negative: 3
    - True Positive: 25
  - K-fold cross validation results: Average test score of 0.91

## References

1. Maddury S. Automated Huntington's Disease Prognosis via Biomedical Signals and Shallow Machine Learning. ArXiv [Preprint]. 2023 Feb 8:arXiv:2302.03605v2.
2. Mohan A., Sun Z., Ghosh S., Li Y., Sathe S., Hu J., Sampaio C. Corrections to “A Machine-Learning Derived Huntington's Disease Progression Model: Insights for Clinical Trial Design”, Movement Disorders, 10.1002/mds.29259, 37, 12, (2468-2468), (2022).
3. Ko J., Furby H., Ma X., Long JD, Lu X-Y, Slowiejko D, Gandhy R. Clustering and prediction of disease progression trajectories in Huntington's disease: An analysis of Enroll-HD data using a machine learning approach. Front. Neurol. 13:1034269. doi: 10.3389/fneur.2022.1034269
4. Brown, M.P. et al. Knowledge-based analysis of microarray gene expression data by using support vector machines. Proceedings of the National Academy of Sciences, 97(1), pp. 262–267. Available at: https://doi.org/10.1073/pnas.97.1.262.
5. Borovecki F., Lovrecic L., Zhou J., Jeong H., Then F., Rosas HD, Hersch SM, Hogarth P., Bouzou B., Jensen RV, Krainc D. Genome-wide expression profiling of human blood reveals biomarkers for Huntington's disease. Proc Natl Acad Sci U S A. 2005 Aug 2;102(31):11023-8. doi: 10.1073/pnas.0504921102
6. Cheng J., Liu HP, Lin WY, Tsai FJ. Identification of contributing genes of Huntington's disease by machine learning. BMC Med Genomics. 2020 Nov 23;13(1):176. doi: 10.1186/s12920-020-00822-w.





