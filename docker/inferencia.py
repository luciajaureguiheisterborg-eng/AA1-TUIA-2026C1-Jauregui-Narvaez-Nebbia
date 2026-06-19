import pandas as pd
import numpy as np
import joblib
import json
import sys

# Cargar artefactos
modelo = joblib.load("modelo.pkl")
scaler = joblib.load("scaler.pkl")
med_loc_month = joblib.load("med_loc_month.pkl")
med_loc_season = joblib.load("med_loc_season.pkl")
med_loc = joblib.load("med_loc.pkl")
med_global = joblib.load("med_global.pkl")
moda_loc_season = joblib.load("moda_loc_season.pkl")
moda_loc = joblib.load("moda_loc.pkl")
moda_global = joblib.load("moda_global.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")  # <-- NUEVO

cols_num = [
    "Sunshine", "Evaporation", "Cloud3pm", "Cloud9am",
    "MinTemp", "MaxTemp", "Rainfall",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm"]

cols_cat_impute = ["WindGustDir", "WindDir9am", "WindDir3pm"]

def imputar_numericas(df):
    for col in cols_num:
        if col not in df.columns:
            continue
        mask = df[col].isnull()
        for idx in df[mask].index:
            loc = df.loc[idx, "Location"]
            month = df.loc[idx, "Month"]
            season = df.loc[idx, "Season"]
            val = np.nan
            if (loc, month) in med_loc_month.index:
                val = med_loc_month.loc[(loc, month), col]
            if pd.isnull(val) and (loc, season) in med_loc_season.index:
                val = med_loc_season.loc[(loc, season), col]
            if pd.isnull(val) and loc in med_loc.index:
                val = med_loc.loc[loc, col]
            if pd.isnull(val):
                val = med_global[col]
            df.loc[idx, col] = val
    return df

def imputar_cat(row, col):
    if pd.notnull(row[col]):
        return row[col]
    key = (row["Location"], row["Season"])
    if key in moda_loc_season.index:
        val = moda_loc_season.loc[key, col]
        if pd.notnull(val):
            return val
    if row["Location"] in moda_loc.index:
        val = moda_loc.loc[row["Location"], col]
        if pd.notnull(val):
            return val
    return moda_global[col]

def preprocesar(df):
    # Fechas
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map({
        12: "Summer", 1: "Summer", 2: "Summer",
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring"})
    df = df.drop(columns=["Date"])

    # Imputar numéricas
    df = imputar_numericas(df)

    # Imputar RainToday
    df["RainToday"] = df["RainToday"].fillna("No")

    # Imputar categóricas
    for col in cols_cat_impute:
        df[col] = df.apply(lambda row: imputar_cat(row, col), axis=1)

    # Encodear direcciones de viento
    direcciones = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                   'S','SSW','SW','WSW','W','WNW','NW','NNW']
    for col in cols_cat_impute:
        df[col + "_sin"] = df[col].map(
            lambda d: np.sin(2*np.pi*direcciones.index(d)/16) if d in direcciones else 0)
        df[col + "_cos"] = df[col].map(
            lambda d: np.cos(2*np.pi*direcciones.index(d)/16) if d in direcciones else 0)
    df = df.drop(columns=cols_cat_impute, errors='ignore')

    # RainToday binario
    df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

    # OHE manual para Season (evita el problema de get_dummies con 1 sola fila)
    season_val = df["Season"].iloc[0]
    df["Season_Spring"] = season_val == "Spring"
    df["Season_Summer"] = season_val == "Summer"
    df["Season_Winter"] = season_val == "Winter"
    df = df.drop(columns=["Season"])

    # Eliminar columnas no necesarias
    df = df.drop(columns=["Location", "Month"], errors='ignore')

    # Reordenar/completar columnas EXACTAMENTE como en el entrenamiento
    df = df.reindex(columns=columnas_modelo, fill_value=0)

    # Escalar manteniendo nombres de columnas
    X = pd.DataFrame(scaler.transform(df), columns=columnas_modelo)
    return X

if __name__ == "__main__":
    data = json.loads(sys.stdin.read())
    df = pd.DataFrame([data])
    X = preprocesar(df)
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]
    resultado = "Llueve" if pred == 1 else "No llueve"
    print(json.dumps({"prediccion": resultado, "probabilidad": round(float(prob), 4)}))