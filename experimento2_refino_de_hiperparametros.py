import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
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


# 2. Função para treinar e buscar o melhor modelo com GridSearch
def train_model_with_gridsearch(X_train, y_train, X_test, y_test):
    """
    Realiza uma busca por hiperparâmetros com GridSearch e treina o melhor modelo.
    Retorna: melhor modelo treinado, precisão, recall, melhor combinação de hiperparâmetros
    """
    # Definir os hiperparâmetros a serem testados no GridSearch
    param_grid = {
        "iterations": [100, 200, 500],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5]
    }

    # Usar ParameterGrid para iterar sobre todas as combinações
    grid = ParameterGrid(param_grid)
    
    best_model = None
    best_precision = 0
    best_recall = 0
    best_params = None

    print(f"Testando {len(grid)} combinações de hiperparâmetros...")

    for params in grid:
        model = CatBoostClassifier(verbose=0, random_seed=42, **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Parâmetros: {params} | Precision: {precision:.4f}, Recall: {recall:.4f}")

        if precision > best_precision and recall > best_recall:
            best_model = model
            best_precision = precision
            best_recall = recall
            best_params = params

    print(f"\nMelhores parâmetros: {best_params}")
    print(f"Melhor Precision: {best_precision:.4f}, Melhor Recall: {best_recall:.4f}")

    return best_model, best_precision, best_recall, best_params


# 3. Função para registrar o modelo no MLflow
def log_model_mlflow(model, precision, recall, version, params, experiment_name="Water_Potability"):
    """
    Registra o modelo no MLflow com as métricas de precisão, recall e hiperparâmetros.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"Model_v{version}"):
        mlflow.catboost.log_model(model, artifact_path="model")
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("version", version)
        print(f"Modelo versão {version} registrado no MLflow com precisão {precision:.4f} e recall {recall:.4f}")


# Script principal
if __name__ == "__main__":
    filepath = "C:/Users/Victor/Desktop/IML4.2/WaterQualityClassifier/data/water_potability.csv"
    print("Carregando os dados...")
    X_train, X_test, y_train, y_test = load_data(filepath)

    print("Treinando o modelo com GridSearch...")
    model, precision, recall, best_params = train_model_with_gridsearch(X_train, y_train, X_test, y_test)

    print("Registrando o melhor modelo no MLflow...")
    log_model_mlflow(model, precision, recall, version=2, params=best_params)
    print("Finalizado!")
