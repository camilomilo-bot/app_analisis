import os
import io
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
CONTAINER_NAME = "mis-archivos"

# Cliente de Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

def subir_df_a_blob(df: pd.DataFrame, blob_name: str) -> bool:
    """Sube un DataFrame a Azure Blob Storage en formato Parquet."""
    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        blob_client.upload_blob(buffer, overwrite=True)
        
        return True
    except Exception as e:
        print(f"Error al subir archivo: {e}")
        return False

def descargar_df_desde_blob(blob_name: str):
    """Descarga un archivo Parquet desde Azure y lo convierte en DataFrame."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        stream = io.BytesIO(blob_client.download_blob().readall())
        
        return pd.read_parquet(stream)
    except Exception as e:
        print(f"Error al descargar archivo: {e}")
        return None
