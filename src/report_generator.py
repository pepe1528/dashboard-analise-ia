from fpdf import FPDF
import os

def gerar_relatorio(df, coluna_x, coluna_y):
    os.makedirs("reports", exist_ok=True)
    caminho_relatorio = f"reports/relatorio_analise.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Relatório de Análise de Dados", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Variável X: {coluna_x}", ln=True, align="L")
    pdf.cell(200, 10, f"Variável Y: {coluna_y}", ln=True, align="L")

    pdf.cell(200, 10, "Resumo Estatístico:", ln=True, align="L")
    pdf.multi_cell(200, 10, str(df.describe()))

    pdf.output(caminho_relatorio)

    return caminho_relatorio