import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Análisis de Arriendos de Bodegas")

# Subir archivo Excel desde la interfaz web
archivo = st.file_uploader("Sube el archivo Excel", type=["xlsx"])

if archivo is not None:
    nombre_hoja = 'OK-DTA V2'
    df = pd.read_excel(archivo, sheet_name=nombre_hoja)

    # Limpieza básica
    df = df[df['Obs'].str.lower().str.contains('arriendo', na=False)]  # Solo arriendos
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Monto'] = pd.to_numeric(df['Monto'], errors='coerce')
    df = df.dropna(subset=['Responsable', 'Monto'])

    st.subheader("Vista previa de datos")
    st.dataframe(df.head())

    # --- Análisis 1: Total por arrendatario ---
    arriendos_totales = df.groupby('Responsable')['Monto'].sum().sort_values(ascending=False)
    st.subheader("Total por arrendatario")
    st.dataframe(arriendos_totales)

    # --- Análisis 2: Evolución mensual ---
    df['Mes-Año'] = df['Fecha'].dt.to_period('M')
    evolucion = df.groupby(['Mes-Año', 'Responsable'])['Monto'].sum().unstack().fillna(0)
    st.subheader("Evolución mensual (últimas filas)")
    st.dataframe(evolucion.tail())

    # --- Análisis 3: Participación por arrendatario (torta) ---
    fig1, ax1 = plt.subplots()
    arriendos_totales.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title('Participación total por arrendatario')
    st.pyplot(fig1)

    # --- Análisis 4: Barras (Top 10 arrendatarios) ---
    fig2, ax2 = plt.subplots()
    arriendos_totales.head(10).plot(kind='bar', ax=ax2)
    ax2.set_title('Top 10 arrendatarios por monto pagado')
    ax2.set_xlabel('Arrendatario')
    ax2.set_ylabel('Monto total arriendo')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # --- Análisis 5: Línea (evolución mensual) ---
    fig3, ax3 = plt.subplots()
    evolucion.sum(axis=1).plot(ax=ax3)
    ax3.set_title("Evolución mensual de arriendos")
    ax3.set_xlabel("Mes-Año")
    ax3.set_ylabel("Monto total")
    st.pyplot(fig3)

else:
    st.info("Por favor, sube el archivo Excel con la hoja OK-DTA V2.")









