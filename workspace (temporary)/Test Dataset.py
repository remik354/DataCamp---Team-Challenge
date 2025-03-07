import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Charger le dataset
file_path = r"C:\Users\cferr\Desktop\M2 Data Science\Data Camp\Group project\student_final.csv"
df = pd.read_csv(file_path, sep=';')

# Définir la target et les features
y = df["G3_por"]
X = df.drop(columns=["G3_por"])

# Identifier les colonnes catégoriques et numériques
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Préprocessing : One-Hot Encoding pour les variables catégoriques, StandardScaler pour les numériques
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])

# Définition des modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel="rbf"),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Boucle pour entraîner et évaluer chaque modèle
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {"MAE": mae, "RMSE": rmse}
    print(f"{name}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

# Affichage des résultats
print("\nRésultats des modèles :")
for model, scores in results.items():
    print(f"{model} -> MAE: {scores['MAE']:.4f}, RMSE: {scores['RMSE']:.4f}")
