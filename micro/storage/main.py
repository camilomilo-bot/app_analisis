from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import pandas as pd
import io
from services.storage_service import subir_df_a_blob, descargar_df_desde_blob

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Microbackend de almacenamiento funcionando"}

@app.post("/subir/")
async def subir_archivo(file: UploadFile = File(...), nombre: str = Query(...)):
    """Sube un archivo Excel a Azure Blob Storage con un nombre personalizado."""
    try:
        df = pd.read_excel(file.file)
        nombre_archivo = f"{nombre}.parquet"  # Se usa el nombre personalizado
        
        if subir_df_a_blob(df, nombre_archivo):
            return {"message": "Archivo subido correctamente", "filename": nombre_archivo}
        else:
            raise HTTPException(status_code=500, detail="Error al subir el archivo")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/descargar/{nombre_archivo}")
def descargar_archivo(nombre_archivo: str):
    """Descarga un archivo desde Azure Blob Storage y lo devuelve en CSV."""
    df = descargar_df_desde_blob(nombre_archivo)
    if df is None:
        raise HTTPException(status_code=404, detail="Archivo no encontrado en Azure")
    
    stream = io.StringIO()
    df.to_csv(stream, index=False, sep=";", decimal=",")
    return {"filename": nombre_archivo, "content": stream.getvalue()}
