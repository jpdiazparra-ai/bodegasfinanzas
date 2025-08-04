import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Cargar datos ---
archivo = 'Proyecto bodegas-2.xlsx'
nombre_hoja = 'OK-DTA V2'
df = pd.read_excel(archivo, sheet_name=nombre_hoja)

# --- Limpieza básica ---
df = df[df['Obs'].str.lower().str.contains('arriendo', na=False)]  # Solo arriendos
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df['Monto'] = pd.to_numeric(df['Monto'], errors='coerce')
df = df.dropna(subset=['Responsable', 'Monto'])

# --- Análisis 1: Total por arrendatario ---
arriendos_totales = df.groupby('Responsable')['Monto'].sum().sort_values(ascending=False)
print(arriendos_totales)

# --- Análisis 2: Evolución mensual ---
df['Mes-Año'] = df['Fecha'].dt.to_period('M')
evolucion = df.groupby(['Mes-Año', 'Responsable'])['Monto'].sum().unstack().fillna(0)
print(evolucion.tail())

# --- Análisis 3: Participación por arrendatario (torta) ---
plt.figure(figsize=(8,6))
arriendos_totales.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Participación total por arrendatario')
plt.ylabel('')
plt.tight_layout()
plt.show()

# --- Análisis 4: Barras (Top 10 arrendatarios) ---
plt.figure(figsize=(10,6))
arriendos_totales.head(10).plot(kind='bar')
plt.title('Top 10 arrendatarios por monto pagado')
plt.xlabel('Arrendatario')
plt.ylabel('Monto total arriendo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Análisis 5: Línea (evolución mensual) ---
plt.figure(figsize=(12,6))
evolucion.sum(axis=1).plot()
plt.title('Evolución mensual de arriendos')
plt.xlabel('Mes-Año')
plt.ylabel('Monto total')
plt.tight_layout()
plt.show()

# --- Análisis 6: Rosa de viento (Frecuencia de pagos por bodega/espacio) ---
if 'Esp' in df.columns:
    # Convertir Esp a string
