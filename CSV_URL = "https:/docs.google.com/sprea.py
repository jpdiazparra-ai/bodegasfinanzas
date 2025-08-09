CSV_URL = "https://docs.google.com/spreadsheets/d/e/.../pub?gid=1154929321&single=true&output=csv"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8-sig")
    df["Fecha"] = pd.to_datetime(df.get("Fecha"), errors="coerce")
    df["Monto"] = pd.to_numeric(
        df["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce"
    )
    # Columnas que usamos después
    for c in ["Obs", "CC1", "Sit", "Responsable", "Año", "Esp"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

try:
    df = load_data(CSV_URL)
except Exception as e:
    st.error(f"No pude leer el CSV de Google Sheets: {e}")
    st.stop()

