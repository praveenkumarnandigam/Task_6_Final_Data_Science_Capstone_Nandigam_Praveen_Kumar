# Final Data Science Capstone Project
## Student Performance Analysis and Pass/Fail Prediction

**Author:** Nandigam Praveen Kumar  
**Internship:** YuvaIntern – Data Analyst  
**Task:** 6 – Final Data Science Capstone Project

### 1. Problem statement
This project explores student performance data and builds a machine-learning model to estimate whether a student will pass the final course assessment. The goal is to demonstrate a complete workflow: problem definition, data collection, cleaning, exploratory analysis, feature preparation, model training, evaluation, interpretation, and recommendations.

### 2. Dataset
The project uses the UCI Machine Learning Repository **Student Performance** dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains student demographic, family, school, and study-related attributes, as well as grades. The notebook retrieves it through the `ucimlrepo` package. No dataset file is bundled; this keeps the repository small and directs users to the original source.

### 3. Important modelling decision
The target is **pass/fail**, derived from final grade `G3` (pass = G3 >= 10 on the dataset's 0–20 scale). Earlier period grades `G1` and `G2` are excluded from model inputs because they are direct prior assessments and can make the prediction unrealistically easy. This project therefore focuses on information available beyond those earlier grades. The model is an educational demonstration, not a tool for making consequential decisions about students.

### 4. Project structure
```text
Task_6_Final_Data_Science_Capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── run_analysis.py
├── notebooks/
│   └── student_performance_capstone.ipynb
├── reports/
│   └── Capstone_Project_Report.docx
├── data/
│   └── README.md
└── outputs/
    └── (generated after running the analysis)
```

### 5. Run the project
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python src/run_analysis.py
   ```
5. Or open `notebooks/student_performance_capstone.ipynb` in Jupyter/Google Colab and run all cells.

The first run requires an internet connection to retrieve the dataset from UCI. The script creates evaluation metrics, charts, a cleaned dataset, and a text summary in `outputs/`.

### 6. Evaluation
The workflow compares a Logistic Regression baseline with a Random Forest classifier. It reports accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC when both classes are present. Use the generated outputs to discuss model performance; do not report metrics before running the notebook/script.

### 7. Limitations and responsible use
- The data comes from two Portuguese secondary schools and may not represent other education systems.
- Associations in observational data do not prove that a factor causes a grade outcome.
- Class balance, sampling, and feature availability can affect the metrics.
- A pass/fail prediction can encode historical inequalities; it should never be used alone to label, punish, or deny opportunities to students.
- Results should be checked on a separate, representative dataset before any real-world use.

### 8. Source
Cortez, P. (2008). *Using data mining to predict secondary school student performance*. UCI Machine Learning Repository, Student Performance dataset. https://archive.ics.uci.edu/dataset/320/student+performance
