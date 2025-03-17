import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

def treinar_modelo(df, coluna_x, coluna_y, modelo_escolhido):
    """Treina o modelo de IA com os dados escolhidos"""
    
    # Verifica se as colunas são numéricas
    if df[coluna_x].dtype not in [np.int64, np.float64] or df[coluna_y].dtype not in [np.int64, np.float64]:
        return "Erro: Selecione colunas numéricas para IA", None, None

    # Preparação dos dados
    X = df[[coluna_x]].values
    y = df[coluna_y].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escolhendo o modelo
    if modelo_escolhido == "Regressão Linear":
        modelo = LinearRegression()
    elif modelo_escolhido == "Árvore de Decisão":
        modelo = DecisionTreeRegressor()
    elif modelo_escolhido == "Random Forest":
        modelo = RandomForestRegressor(n_estimators=100)
    elif modelo_escolhido == "KNN":
        modelo = KNeighborsRegressor(n_neighbors=5)
    elif modelo_escolhido == "Regressão Logística":
        modelo = LogisticRegression()
    else:
        return "Erro: Modelo não reconhecido", None, None

    # Treinar modelo
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Calcular erro médio absoluto
    erro = mean_absolute_error(y_test, y_pred)

    return modelo, erro, (X_test, y_pred)