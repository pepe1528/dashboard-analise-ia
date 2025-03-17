import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from report_generator import gerar_relatorio

# Configuração da página
st.set_page_config(page_title="Dashboard de Análise de Dados", layout="wide")

# Título do dashboard
st.title("📊 Dashboard de Análise de Dados com IA")

# Upload de arquivo
st.sidebar.subheader("Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dados carregados com sucesso!")

    # Exibir os dados
    st.subheader("📂 Visualização dos Dados")
    st.write(df.head())

    # Gráficos interativos
    st.subheader("📈 Gráficos Interativos")
    coluna_x = st.selectbox("Escolha a variável para o eixo X", df.columns)
    coluna_y = st.selectbox("Escolha a variável para o eixo Y", df.columns)

    fig = px.scatter(df, x=coluna_x, y=coluna_y, title="Relação entre Variáveis")
    st.plotly_chart(fig)

    # Escolha do modelo de IA
    st.subheader("🤖 Previsão com Inteligência Artificial")

    modelos_ia = ["Regressão Linear", "Árvore de Decisão", "Random Forest", "KNN", "Regressão Logística"]
    modelo_escolhido = st.selectbox("Escolha um modelo de IA", modelos_ia)

    if df[coluna_x].dtype in [np.int64, np.float64] and df[coluna_y].dtype in [np.int64, np.float64]:
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

        # Treinar o modelo
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Cálculo do erro
        erro = mean_absolute_error(y_test, y_pred)

        # Criar previsões
        previsao_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        previsao_y = modelo.predict(previsao_x)

        # Exibir resultados
        st.write(f"Modelo escolhido: **{modelo_escolhido}**")
        st.write(f"Erro Médio Absoluto (MAE): **{erro:.4f}**")

        # Plotar a previsão
        fig_ai = px.scatter(df, x=coluna_x, y=coluna_y, title=f"Modelo: {modelo_escolhido}")
        fig_ai.add_scatter(x=previsao_x.flatten(), y=previsao_y, mode="lines", name="Previsão")
        st.plotly_chart(fig_ai)

        # Botão para gerar relatório em PDF
        if st.button("Gerar Relatório PDF"):
            caminho_relatorio = gerar_relatorio(df, coluna_x, coluna_y)
            st.success(f"Relatório salvo em: {caminho_relatorio}")

    else:
        st.warning("Selecione variáveis numéricas para previsão.")

else:
    st.info("Por favor, faça o upload de um arquivo CSV para continuar.")