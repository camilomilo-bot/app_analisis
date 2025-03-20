from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import io
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import gc

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
        # Cargar DataFrame con optimización de tipos
        df = pd.read_parquet(stream)
        df = optimizar_dataframe(df)  # Optimizar el uso de memoria
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al descargar archivo: {e}")

def subir_df_a_blob(df, blob_name):
    """Sube un DataFrame en formato Parquet a Azure Blob Storage"""
    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(buffer, overwrite=True)

        # Liberar memoria
        del df
        gc.collect()


        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir archivo: {e}")

@app.get("/")
def home():
    return {"message": "Microbackend de data base y com_nits funcionando"}

@app.post("/crear_base_principal/")
async def crear_base_principal():
    """Genera una base principal sin NITs repetidos y la guarda en Azure."""
    try:
        df_nits = descargar_df_desde_blob("BaseSecundaria.parquet")
        df_datos = descargar_df_desde_blob("BasePrincipal.parquet")

        # Filtrar los NITs que no estén en la base secundaria
        df_sin_nits = df_datos[~df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

        if subir_df_a_blob(df_sin_nits, "BasePrincipalSNIT.parquet"):
            response = {"message": "Base Principal actualizada correctamente"}
        else:
            raise HTTPException(status_code=500, detail="Error al subir la base principal")
        
         # Liberar memoria
        del df_nits, df_datos, df_sin_nits
        gc.collect()

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/completar_nits/")
async def completar_nits():
    """Completa la base de datos con NITs a partir de una base temporal."""
    try:
        df_nits = descargar_df_desde_blob("temporal.parquet")
        df_datos = descargar_df_desde_blob("BasePrincipal.parquet")

        if df_nits is None or df_datos is None:
            raise HTTPException(status_code=500, detail="No se pudieron descargar los datos")

        if "NIT" in df_nits.columns:
            df_nits.rename(columns={"NIT": "IDENTIFICACION"}, inplace=True)

        if "IDENTIFICACION" not in df_datos.columns:
            raise HTTPException(status_code=500, detail="Columna IDENTIFICACION no encontrada en base principal")

        df_filtrado = df_datos[df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

        if subir_df_a_blob(df_filtrado, "BaseSecundaria.parquet"):
            response = {"message": "Base Secundaria actualizada correctamente"}
        else:
            raise HTTPException(status_code=500, detail="Error al subir la base secundaria")
        

        # Liberar memoria
        del df_nits, df_datos, df_filtrado
        gc.collect()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
