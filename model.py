import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

print("=" * 60)
print("Mileage Predictor – Random Forest Training")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("cars_data.csv")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# ---------------------------------------------------------
# 2. TARGET CLEANING
# ---------------------------------------------------------

df["Mileage"] = df["ARAI_Certified_Mileage"].str.extract(r"([\d.]+)").astype(float)
df = df.dropna(subset=["Mileage"])
df = df[(df["Mileage"] >= 5) & (df["Mileage"] <= 60)]

# ---------------------------------------------------------
# 3. NUMERIC CLEANING
# ---------------------------------------------------------

df["Displacement_cc"] = df["Displacement"].str.extract(r"([\d,.]+)").replace(",", "", regex=True).astype(float)
df["Kerb_Weight_kg"]  = df["Kerb_Weight"].str.extract(r"([\d,.]+)").replace(",", "", regex=True).astype(float)
df["Power_val"]       = df["Power"].str.extract(r"([\d,.]+)").replace(",", "", regex=True).astype(float)
df["Torque_val"]      = df["Torque"].str.extract(r"([\d,.]+)").replace(",", "", regex=True).astype(float)

df["Gears"] = pd.to_numeric(df["Gears"], errors="coerce")
df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")

df = df[(df["Displacement_cc"] >= 500) & (df["Displacement_cc"] <= 8000)]

# ---------------------------------------------------------
# 4. STANDARDIZE DRIVETRAIN
# ---------------------------------------------------------

df["Drivetrain"] = df["Drivetrain"].str.strip()

df["Drivetrain"] = df["Drivetrain"].replace({
    "AWD": "4WD",
    "AWD/4WD": "4WD",
    "4WD/AWD": "4WD",
    "AWD - 4WD": "4WD",
    "4WD - AWD": "4WD"
})

# ---------------------------------------------------------
# 5. LABEL ENCODING
# ---------------------------------------------------------

cat_cols = ["Fuel_Type", "Drivetrain", "Body_Type"]

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ---------------------------------------------------------
# 6. FEATURE SELECTION
# ---------------------------------------------------------

FEATURE_COLS = [
    "Displacement_cc",
    "Kerb_Weight_kg",
    "Power_val",
    "Torque_val",
    "Gears",
    "Cylinders",
    "Fuel_Type",
    "Drivetrain",
    "Body_Type",
]

X = df[FEATURE_COLS]
y = df["Mileage"].values

# ---------------------------------------------------------
# 7. HANDLE MISSING VALUES
# ---------------------------------------------------------

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

# ---------------------------------------------------------
# 8. TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 9. RANDOM FOREST MODEL
# ---------------------------------------------------------

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ---------------------------------------------------------
# 10. MODEL EVALUATION
# ---------------------------------------------------------

preds = rf.predict(X_test)

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)

print(f"\nSamples used : {len(X)}")
print(f"MAE          : {mae:.3f} km/l")
print(f"RMSE         : {rmse:.3f} km/l")
print(f"R² (Test)    : {r2:.4f}")

# ---------------------------------------------------------
# 11. SAVE ONLY RF MODEL
# ---------------------------------------------------------

joblib.dump(rf, "rf_model.pkl")

print("\nOnly rf_model.pkl saved successfully.")
print("=" * 60)