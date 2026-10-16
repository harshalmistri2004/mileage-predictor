# Car Mileage Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Machine_Learning-Random_Forest-2E8B57?style=for-the-badge" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
</p>

<p align="center">
  <b>Predict smarter. Drive efficiently.</b>
  <br/>
  A machine learning web application that estimates car fuel efficiency from vehicle specifications.
</p>

<p align="center">
  <a href="https://github.com/harshalmistri2004/mileage-predictor">View Repository</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/harshalmistri2004/mileage-predictor/issues">Report an Issue</a>
</p>

---

## Overview

**Car Mileage Predictor** is a machine learning project built with Python and Streamlit. It uses a trained Random Forest regression model to estimate vehicle mileage based on engine specifications and vehicle characteristics.

The application provides an interactive interface where users can adjust vehicle parameters, generate a mileage prediction, and view information about the underlying model.

## Features

| Feature | Description |
|---|---|
| Interactive Prediction | Enter vehicle specifications through Streamlit controls. |
| Fuel Type Selection | Select Petrol, Diesel, or CNG. |
| Vehicle Configuration | Choose body type and drivetrain. |
| Engine Specifications | Configure displacement, cylinders, power, and torque. |
| Vehicle Parameters | Adjust kerb weight and number of gears. |
| Machine Learning | Generate predictions using a trained Random Forest Regressor. |
| Model Information | View the model algorithm and displayed performance information. |

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn
- **Model:** Random Forest Regressor
- **Data Processing:** NumPy, Pandas
- **Model Persistence:** Joblib
- **Dataset:** `cars_data.csv`

## How It Works

```text
Vehicle Specifications
          |
          v
Feature Encoding
          |
          v
Trained Random Forest Model
          |
          v
Mileage Prediction
          |
          v
Result Displayed in Streamlit
```

The application converts selected categorical inputs into numerical values, combines them with numeric vehicle specifications, and passes the resulting feature vector to the trained model.

## Project Structure

```text
mileage-predictor/
│
├── app.py              # Streamlit web application
├── model.py            # Model development and training code
├── cars_data.csv       # Vehicle dataset
├── rf_model.pkl        # Serialized trained model
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/harshalmistri2004/mileage-predictor.git
cd mileage-predictor
```

### 2. Create a Virtual Environment

**Windows — Git Bash**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the packages used by the project:

```bash
python -m pip install streamlit pandas numpy scikit-learn joblib
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, typically:

`http://localhost:8501`

## How to Use

1. Launch the Streamlit application.
2. Open the **Predict** tab.
3. Select the fuel type, body type, and drivetrain.
4. Adjust engine displacement, cylinders, power, torque, kerb weight, and gears.
5. Click **Predict Mileage**.
6. Review the estimated mileage in kilometres per litre.
7. Open **Model Information** to view the displayed model details.

## Model Information

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Problem Type | Regression |
| Number of Trees | 300, as displayed by the application |
| Output | Estimated mileage in km/L |

The application currently displays an R² value of `0.95`. This value should be independently verified against evaluation results on a held-out test dataset before being treated as validated model performance.

## Future Improvements

- [ ] Add model evaluation metrics such as MAE, RMSE, and R² from a test dataset.
- [ ] Improve input validation and error handling.
- [ ] Add visualizations for feature importance.
- [ ] Compare Random Forest with other regression algorithms.
- [ ] Add automated tests for prediction logic.
- [ ] Deploy the application for public access.

## Best Practices

- Keep the Python virtual environment out of Git.
- Do not commit generated cache files.
- Keep model training and inference dependencies consistent.
- Document the dataset source and licensing before redistribution.
- Validate model performance before using predictions for real-world decisions.

## Author

**Harshal Mistri**

GitHub: [@harshalmistri2004](https://github.com/harshalmistri2004)

---

<p align="center">
  <b>Built with Python, Machine Learning, and Streamlit.</b>
  <br/>
  If you find this project useful, consider giving the repository a star.
</p>

<p align="center">
  <sub>Predictions are estimates and may differ from real-world fuel economy.</sub>
</p>
