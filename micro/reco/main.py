from fastapi import FastAPI, HTTPException, Response
import pandas as pd
import numpy as np
import os
import io
from dotenv import load_dotenv
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from azure.storage.blob import BlobServiceClient
import re
import gc  # Garbage Collector

# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI()

# Conexión a Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))
container_name = "mis-archivos"


def optimizar_dataframe(df):
    """Convierte tipos de datos para reducir el uso de memoria"""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")  # Reducir precisión flotante

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")  # Reducir tamaño de enteros

    return df


def descargar_df_desde_blob(blob_name):
    """Descarga un archivo Parquet desde Azure y lo convierte en un DataFrame"""
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        stream = io.BytesIO(blob_client.download_blob().readall())
        df = pd.read_parquet(stream)
        df = optimizar_dataframe(df)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al descargar archivo: {e}")

def extraer_numeros_ciiu(codigo):
    """Extrae solo los números del código CIIU"""
    if isinstance(codigo, str):
        match = re.search(r'\d+', codigo)
        return match.group(0) if match else None
    return None

def modelo_principal_sec():
    """Ejecuta el modelo de recomendaciones y devuelve un DataFrame con los resultados."""
    try:
        base_secundaria = descargar_df_desde_blob("BaseSecundaria.parquet")
        base_principal = descargar_df_desde_blob("BasePrincipalSNIT.parquet")

        # Convertir nombres de columnas a minúsculas
        base_secundaria = base_secundaria.rename(columns=str.lower)
        base_principal = base_principal.rename(columns=str.lower)

        for col in ["patrimonio", "personal"]:
            base_secundaria[col] = pd.to_numeric(base_secundaria[col], errors="coerce").fillna(0)
            base_principal[col] = pd.to_numeric(base_principal[col], errors="coerce").fillna(0)

        scaler = RobustScaler()
        clientes_base_secundaria = scaler.fit_transform(base_secundaria[["patrimonio", "personal"]])
        clientes_base_principal = scaler.transform(base_principal[["patrimonio", "personal"]])

        distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="euclidean")
        mejores_indices = np.argmin(distancias, axis=1)

        resultados = pd.DataFrame({
            "Identificacion": base_principal["identificacion"],
            "EMPRESA": base_principal["empresa"],
            "Patrimonio": base_principal["patrimonio"],
            "Personal": base_principal["personal"],
            "Codigo_CIIU": base_principal["codigo_ciiu"],
            "Distancia": distancias[np.arange(len(base_principal)), mejores_indices],
            "Identificacion_cliente_1": base_secundaria.iloc[mejores_indices]["identificacion"].values,
            "EMPRESA_Cliente_1": base_secundaria.iloc[mejores_indices]["empresa"].values,
            "Patrimonio_cliente_1": base_secundaria.iloc[mejores_indices]["patrimonio"].values,
            "Personal_cliente_1": base_secundaria.iloc[mejores_indices]["personal"].values,
            "Codigo_CIIU_Cliente_1": base_secundaria.iloc[mejores_indices]["codigo_ciiu"].values,
        })

        resultados["Distancia"] = resultados["Distancia"].round(4)

        # Filtrado por Código CIIU
        codigos_ciiu_secundaria = set(
            base_secundaria["codigo_ciiu"].dropna().apply(extraer_numeros_ciiu).dropna().unique()
        )

        resultados["Codigo_CIIU_Numerico"] = resultados["Codigo_CIIU"].apply(extraer_numeros_ciiu)
        resultados = resultados[resultados["Codigo_CIIU_Numerico"].isin(codigos_ciiu_secundaria)]
        resultados = resultados.drop(columns=["Codigo_CIIU_Numerico"])

        # Liberar memoria
        del base_secundaria, base_principal, distancias, mejores_indices
        gc.collect()


        return resultados
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar modelo: {e}")

@app.get("/")
def home():
    return {"message": "Microbackend de reconomendacion funcionando"}

@app.get("/generar_recomendaciones/")
async def generar_recomendaciones():
    """Ejecuta el modelo y devuelve un archivo CSV con las recomendaciones."""
    try:
        resultados = modelo_principal_sec()

        if resultados.empty:
            raise HTTPException(status_code=400, detail="No se generaron recomendaciones.")

        # Convertir a CSV en memoria
        buffer = io.StringIO()
        resultados.to_csv(buffer, index=False, sep=";", decimal=",")
        buffer.seek(0)

        # Liberar memoria
        del resultados
        gc.collect()

        return Response(content=buffer.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=recomendaciones.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
