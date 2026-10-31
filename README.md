# Research Abstract Classification Using Machine Learning

## Live Demo

**Try the application here:**  
https://abstractclassifierproject-6kohxukm83kclmxd5wjpyv.streamlit.app/

## Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) techniques to classify research paper abstracts into three categories:

- Science
- Business
- Social Science

The project compares multiple machine learning algorithms to identify the best-performing model for research abstract classification.

## Dataset

The dataset combines three separate CSV files:

- `science.csv`
- `business.csv`
- `socialScience.csv`

After preprocessing, duplicate removal, and removal of ambiguous records, the final dataset contains **5,418 research abstracts**.

### Dataset Distribution

| Category       | Number of Abstracts |
| -------------- | ------------------: |
| Science        |               1,984 |
| Business       |               1,721 |
| Social Science |               1,713 |
| **Total**      |           **5,418** |

The `Abstract` column is used as the primary text input, and `label` is used as the target variable.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Algorithms

The following six machine learning algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. Naive Bayes
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Random Forest

## Model Performance

The models were evaluated using accuracy, precision, recall, and weighted F1-score.

| Algorithm              | Accuracy | Weighted F1-Score |
| ---------------------- | -------: | ----------------: |
| Logistic Regression    |   94.19% |            94.19% |
| Support Vector Machine |   94.10% |            94.08% |
| K-Nearest Neighbors    |   92.44% |            92.41% |
| Naive Bayes            |   92.44% |            92.35% |
| Random Forest          |   91.88% |            91.84% |
| Decision Tree          |   77.31% |            77.45% |

**Best Performing Model: Logistic Regression**

Logistic Regression achieved the highest accuracy of approximately 94.19%.

## Project Structure

```text
abstract_classifier_project/
│
├── data/
│   └── abstracts_labeled.csv
│
├── notebooks/
│   └── abstract_classification_exploration.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── train.py
│
├── models/
│   ├── best_model.joblib
│   └── model_info.json
│
├── reports/
│   ├── model_comparison.csv
│   ├── classification_report_*.csv
│   ├── confusion_matrix_*.png
│   ├── test_class_counts.csv
│   └── dataset_summary.txt
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone or Download the Project

Download the project ZIP file and extract it.

Open Terminal and navigate to the project directory:

```bash
cd abstract_classifier_project
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

## Running the Project

### Run the Streamlit Web Application

```bash
streamlit run app.py
```

The application will open in your browser.

Alternatively, access the deployed application directly:

https://abstractclassifierproject-6kohxukm83kclmxd5wjpyv.streamlit.app/

### Train the Machine Learning Models Again

```bash
python src/train.py
```

This trains all six machine learning models, evaluates their performance, generates classification reports and confusion matrices, and saves the best-performing model.

## Application Features

The Streamlit application provides:

- Model performance comparison.
- Accuracy and evaluation metrics.
- Confusion matrix visualization.
- Dataset class distribution.
- Single research abstract classification.
- Batch classification using CSV file upload.
- Downloadable CSV containing predicted labels.

### Single Abstract Prediction

Users can paste a research abstract or upload a `.txt` file to predict its category
