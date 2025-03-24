

# 📊 Dashboard de Análise de Dados com IA


Este é um **dashboard interativo** desenvolvido em **Python** usando **Streamlit**, que permite:  
✅ **Upload de arquivos CSV** 📂  
✅ **Visualização de dados** em tabelas e gráficos 📈  
✅ **Previsão de tendências** usando **Machine Learning** 🤖  
✅ **Geração de relatórios automáticos em PDF** 📑  

---

## ⚙ **Tecnologias Utilizadas**
- **Python** 🐍
- **Streamlit** 🖥
- **Pandas & Numpy** 📊
- **Matplotlib & Plotly** 🎨
- **Scikit-Learn (Machine Learning)** 🤖
- **FPDF (Geração de Relatórios PDF)** 📄

---
=======
##  Tecnologias Utilizadas
- **Python**
- **Streamlit**
- **Pandas**
- **Matplotlib & Plotly**
- **Scikit-Learn (Machine Learning)**
- **FPDF (Relatórios PDF)**


## 📂 Estrutura do Projeto

dashboard-analise-ia/ │── src/
│ ├── app.py # Arquivo principal do dashboard
│ ├── ai_models.py # Modelos de IA
│ ├── report_generator.py # Geração de relatórios PDF
│── data/ # Arquivos de dados (CSV, Excel)
│── reports/ # Relatórios gerados
│── requirements.txt # Lista de dependências
│── README.md # Documentação

### 🛠 **Como Instalar e Rodar o Projeto**
### **1️⃣ Clonar o Repositório**
```bash
git clone https://github.com/pedroocastilho/dashboard-analise-ia.git
cd dashboard-analise-ia


2️⃣ Criar e Ativar um Ambiente Virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate      # Windows (PowerShell)



3️⃣ Instalar as Dependências
pip install -r requirements.txt

---

4️⃣ Rodar o Dashboard
streamlit run src/app.py

Agora o dashboard será aberto no navegador!

Funcionalidades
✅ Carregar Arquivo CSV: Faça o upload de um dataset e visualize os dados.
✅ Gráficos Interativos: Visualize tendências e padrões com Plotly.
✅ Escolha de Modelos de IA: Compare diferentes algoritmos de Machine Learning.
✅ Geração de Relatórios PDF: Exporte análises automaticamente em PDF.

---

Modelos de IA Disponíveis
O dashboard utiliza vários modelos de Machine Learning para previsão, incluindo:

Modelo de IA	Descrição
Regressão Linear	Modelo simples para prever relações entre variáveis.
Árvore de Decisão	Método baseado em divisão hierárquica dos dados.
Random Forest	Conjunto de várias árvores de decisão para melhorar previsões.
KNN (K-Nearest Neighbors)	Algoritmo baseado na proximidade dos pontos de dados.
Regressão Logística	Modelo usado para previsões categóricas e probabilísticas.


Possíveis Melhorias Futuras
🔹 Exportação dos gráficos como imagem 
🔹 Suporte para mais formatos de arquivo 
🔹 Implementação de Redes Neurais para previsões avançadas 


 Contato
Caso tenha alguma dúvida, entre em contato:
📩 Email: pedrocastilho15@hotmail.com.br
🔗 GitHub: https://github.com/pedroocastilho
🔗 LinkedIn: https://www.linkedin.com/in/pedro-castilho-b03120356/

Se gostou do projeto, deixe uma ⭐ no repositório!