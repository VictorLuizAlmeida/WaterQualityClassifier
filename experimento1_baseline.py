import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import mlflow
import mlflow.catboost


# 1. Função para carregar e dividir os dados
def load_data(filepath):
    """
    Carrega o dataset e divide em treino e teste.
    - filepath: Caminho do arquivo CSV.
    Retorna: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(filepath)
    df = df.dropna()
    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Função para treinar o modelo
def train_model(X_train, y_train, X_test, y_test):
    """
    Treina um modelo CatBoostClassifier e calcula precisão e recall.
    Retorna: modelo treinado, precisão, recall
    """
    model = CatBoostClassifier(verbose=0, random_seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return model, precision, recall


# 3. Função para registrar o modelo no MLflow
def log_model_mlflow(model, precision, recall, version, experiment_name="Water_Potability"):
    """
    Registra o modelo no MLflow com as métricas de precisão e recall.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"Model_v{version}"):
        mlflow.catboost.log_model(model, artifact_path="model")
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("version", version)
        print(f"Modelo versão {version} registrado no MLflow com precisão {precision:.4f} e recall {recall:.4f}")

if __name__ == "__main__":
    filepath = "C:/Users/Victor/Desktop/IML4.2/WaterQualityClassifier/data/water_potability.csv"
    print("Carregando os dados...")
    X_train, X_test, y_train, y_test = load_data(filepath)
    model, precision, recall = train_model(X_train, y_train, X_test, y_test)
    log_model_mlflow(model, precision, recall, version=1)
