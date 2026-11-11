# EV Charging Duration Prediction

Predicting electric vehicle charging duration using exploratory data analysis and a Linear & Logistic Regression model.

## Team Members and Roles

| Team Member ID | Entry Number |Role and Contribution | 
|---|---|---|
| anika-s40 | 2026AIB1013 | Documentation Lead |
| Samarthgupta1406 | 2026EEB1406 | Video Lead|
| Aryan-iitrpr | 2026EEB1327 | Execution Lead |
| SaurabhDagar29 | 2026MCB1566 | PPT Lead |

All the group members have also contributed to the coding part of the model.

## Problem Statement

Electric vehicle charging duration can depend on factors such as battery capacity, charging rate, starting state of charge, vehicle model, and environmental conditions.
The objective of this project is to explore an EV charging dataset and build a regression model to predict **Charging Duration (hours)** from selected vehicle, charging, and environmental features.

## Dataset

- **File:** `data/ev_charging_patterns.csv`
- **Rows:** 1,320
- **Columns:** 20
- **Target variable:** `Charging Duration (hours)`
- **Dataset source:** [Electric Vehicle Charging Patterns — Kaggle](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)

The dataset contains vehicle details, battery capacity, charging rate, charging duration, energy consumed, charging cost, time information, state of charge, temperature, charger type, and user type.

## Project Structure

```text
ev-charging-duration-prediction/
├── data/
│   └── ev_charging_patterns.csv
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── duration_distribution.png
│   ├── evaluation_metrics.csv
│   ├── logistic_confusion_matrix.png
│   └── residual_plot.png
├── ev_charging_prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3
- pip

The Python libraries used include NumPy, pandas, Matplotlib, Seaborn, and scikit-learn.

## Methodology

1. Load and inspect the dataset.
2. Calculate descriptive statistics (mean,median etc.) and examine correlations.
3. Visualize the distribution of charging duration using graphs and Select features.
4. Handle missing numerical and categorical values using preprocessing pipelines.
5. Split the data into training and test sets.
6. Train a Linear Regression model.
7. Compare it with a mean-prediction baseline and evaluate the predictions using MAE, RMSE, and R².
8. Train a Logistic Regression Model.
9. Generate Confusion Matrix for the Model and other evaluation parameters.
10. Save the evaluation metrics and visualizations.

## Setup and Run

Clone the repository:

```bash
git clone https://github.com/Aryan-iitrpr/ev-charging-duration-prediction.git
cd ev-charging-duration-prediction
```

Create and activate a virtual environment :
for macos\linux users:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
for windows users
```bash
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the project:

```bash
python ev_charging_prediction.py
```

The script reads the dataset from the repository's `data/` folder, performs analysis, trains and evaluates the models, and saves output files in the `outputs/` folder.

## Expected Output

After running the project, the following files will be generated in the `outputs/` folder:

- `actual_vs_predicted.png` — Actual versus predicted charging durations.
- `correlation_heatmap.png` — Correlation Heatmap for the model.
- `duration_distribution.png` — Distribution of charging duration.
- `evaluation_metrics.csv` — Evaluation metrics for the baseline and Linear Regression model.
- `logistic_confusion_matrix.png` - Confusion Matrix for the Logistic Regression model.
- `residual_plot.png` — Residual plot for the Linear Regression model.

### Actual vs Predicted Charging Duration

![Actual vs predicted charging duration](outputs/actual_vs_predicted.png)
 

## Results

The current test-set results are:

| Model | MAE (hours) | RMSE (hours) | R² |
|---|---:|---:|---:|
| Mean baseline | 0.8579 | 1.0754 | approximately 0.0000 |
| Linear Regression | 0.8726 | 1.0996 | -0.0455 |

The Linear Regression model performed slightly worse than the mean baseline on the test set. Its negative R² indicates that, on this split, it performed worse than predicting the mean target value.
The Logistic Regression model performs similarly.

These results suggest that the selected features and linear model do not yet provide strong predictive performance. Further feature analysis, data-quality checks, and experiments with other models may improve the results.

## Output Files

- `duration_distribution.png`: Distribution of charging duration.
- `correlation_heatmap.png`: Correlation Heatmap.
- `actual_vs_predicted.png`: Actual versus predicted charging durations on the test set.
- `residual_plot.png`: Residuals against predicted values.
- `logistic_confusion_matrix.png`: Confusion Matrix for Logistic Regression Model.
- `evaluation_metrics.csv`: Model evaluation metrics.

## Presentation and Demo

- **Slides:** https://drive.google.com/file/d/1JqACy03oN0GLFryTHKoQ1qEY3RaUmhI1/view?usp=drive_link
- **Demo Video:** https://drive.google.com/file/d/1XA0q8qIhAUd6r-FlE32NGtW8u2gHiRE8/view?usp=sharing

## AI-Use Disclosure

AI assistance was used for guidance on project development, debugging, explanations, and documentation. The code was run and the reported results were checked against the local program output. Ai was also used for learning new syntax used in the python file. It assisted members to get information about editing videos and making better presentations. 

## Limitations and Future Work

- The current Linear or Logistic Regression model does not outperform the mean baseline.
- Investigate relationships between selected features and charging duration.
- Check for data-quality issues and potentially informative features.
- Experiment with other regression models and compare their test-set performance.

## Reproducibility

The project uses a fixed random state for the train-test split. For consistent reproduction across machines, the Python and library versions should also be documented and pinned.
