import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from azure.storage.blob import BlobServiceClient
import re
import io

load_dotenv()

df_principal = None
df_secundaria = None

blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION_STRING"))

container_name = "mis-archivos"  

def extraer_numeros_ciiu(codigo):
    """ Extrae solo los números del código CIIU """
    if isinstance(codigo, str):
        match = re.search(r'\d+', codigo)  # Busca la primera secuencia numérica
        return match.group(0) if match else None
    return None


def subir_df_a_blob(df, blob_name):
    """Convierte un DataFrame a Parquet y lo sube directamente a Azure Blob Storage"""
    try:
        # Convertir DataFrame a formato Parquet en memoria
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)  # Volver al inicio del buffer
        # Subir a Azure Blob Storage
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(buffer, overwrite=True)
        return True
    except Exception as e:
        st.error(f"Error al subir archivo: {e}")
        return False

def descargar_df_desde_blob(blob_name):
    """Descarga un archivo Parquet desde Azure y lo convierte en un DataFrame"""
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        stream = io.BytesIO(blob_client.download_blob().readall())

        # Convertir de Parquet a DataFrame
        df = pd.read_parquet(stream)
        

        return df
    except Exception as e:
        st.error(f"Error al descargar archivo: {e}")
        return None


def cargar_archivo(uploaded_file, filename):
    """Lee un archivo Excel, lo convierte a Parquet en memoria y lo sube a Azure Blob Storage."""
    if not uploaded_file:
        return False
    try:
        df = pd.read_excel(uploaded_file, dtype=str)
        # Subir a Azure Blob Storage
        subir_df_a_blob(df, filename)  # Asegúrate de que esta función acepte un BytesIO

        return True
    except Exception as e:
        print(f"Error al subir el archivo: {e}")
        return False
    

def crear_base_principal():
    try:
        # Cargar archivos con Dask en formato Parquet
        df_nits = descargar_df_desde_blob(blob_name="BaseSecundaria.parquet")
        df_datos = descargar_df_desde_blob(blob_name="BasePrincipal.parquet")

        # Filtrar los NITs presentes en la base grande
        df_sin_nits = df_datos[~df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

        # Subir a Azure Blob Storage
        result = subir_df_a_blob(df_sin_nits, blob_name="BasePrincipalSNIT.parquet")

        return result  # True si se subió correctamente, False si hubo un error

    except Exception as e:
        print(f"Error en crear_base_principal: {e}")
        return False


def completar_nits():
    # Descargar los DataFrames desde Azure Blob
    df_nits = descargar_df_desde_blob(blob_name="temporal.parquet")
    df_datos = descargar_df_desde_blob(blob_name="BasePrincipal.parquet")

    # Verificar que los DataFrames no estén vacíos
    if df_nits is None or df_datos is None:
        print("Error: No se pudieron descargar los datos desde el blob")
        return

    # Renombrar la columna "NIT" a "IDENTIFICACION" si existe
    if "NIT" in df_nits.columns:
        df_nits.rename(columns={"NIT": "IDENTIFICACION"}, inplace=True)

    # Verificar que la columna "IDENTIFICACION" existe en df_datos
    if "IDENTIFICACION" not in df_datos.columns:
        print("Error: La columna IDENTIFICACION no existe en df_datos")
        return

    # Filtrar los NITs presentes en el archivo grande
    df_filtrado = df_datos[df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

    # Subir el archivo filtrado a Azure Blob Storage
    result = subir_df_a_blob(df=df_filtrado, blob_name="BaseSecundaria.parquet")

def modelo_principal_sec():
  
    base_secundaria = descargar_df_desde_blob("BaseSecundaria.parquet")
    base_principal = descargar_df_desde_blob("BasePrincipalSNIT.parquet")

    # 📌 Convertir nombres de columnas a minúsculas
    base_secundaria = base_secundaria.rename(columns=str.lower)
    base_principal = base_principal.rename(columns=str.lower)


    for col in ["patrimonio", "personal"]:
        base_secundaria[col] = pd.to_numeric(base_secundaria[col], errors="coerce").fillna(0)
        base_principal[col] = pd.to_numeric(base_principal[col], errors="coerce").fillna(0)
    
    scaler = RobustScaler()
    clientes_base_secundaria = scaler.fit_transform(base_secundaria[["patrimonio", "personal"]])
    clientes_base_principal = scaler.transform(base_principal[["patrimonio", "personal"]])
    
    distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="euclidean")
    # Obtener los tres mejores índices ordenados por menor distancia
    mejores_indices = np.argmin(distancias, axis=1)
    
    resultados = pd.DataFrame({
        "Identificacion": base_principal["identificacion"],
        "EMPRESA": base_principal["empresa"],
        "Patrimonio": base_principal["patrimonio"],
        "Personal": base_principal["personal"],
        "Codigo_CIIU": base_principal["codigo_ciiu"],
        "Distancia": distancias[np.arange(len(base_principal)), mejores_indices],

        # Mejor oferta
        "Identificacion_cliente_1": base_secundaria.iloc[mejores_indices]["identificacion"].values,
        "EMPRESA_Cliente_1": base_secundaria.iloc[mejores_indices]["empresa"].values,
        "Patrimonio_cliente_1": base_secundaria.iloc[mejores_indices]["patrimonio"].values,
        "Personal_cliente_1": base_secundaria.iloc[mejores_indices]["personal"].values,
        "Codigo_CIIU_Cliente_1": base_secundaria.iloc[mejores_indices]["codigo_ciiu"].values,
        
        })
        
    resultados["Distancia"] = resultados["Distancia"].round(4)

    #Filtro de codigo_CIUU
        
    codigos_ciiu_secundaria = set(
            base_secundaria["codigo_ciiu"].dropna().apply(extraer_numeros_ciiu).dropna().unique()
        )

        # Extraer los números de los códigos CIIU de la base principal antes de filtrar
    resultados["Codigo_CIIU_Numerico"] = resultados["Codigo_CIIU"].apply(extraer_numeros_ciiu)
        
        # Filtrar donde el código CIIU numérico de base_principal esté en los códigos de base_secundaria
    resultados = resultados[resultados["Codigo_CIIU_Numerico"].isin(codigos_ciiu_secundaria)]
        
        # Eliminar la columna auxiliar después del filtrado
    resultados = resultados.drop(columns=["Codigo_CIIU_Numerico"])

    return resultados

st.sidebar.title("Menú")
opcion = st.sidebar.radio("Seleccione una opción", ["Recomendaciones", "Actualizar Base Principal"])

if opcion == "Recomendaciones":
    st.title("Generador de Recomendaciones")
    uploaded_file = st.file_uploader("Sube la base con NIT o Identificación", type=["xlsx", "xls"])

    if uploaded_file:

        if st.button("Subir Archivo"):
            start_time = time.time()
            with st.status("Cargando datos... por favor espera.", expanded=True) as status:
                st.write("Paso 1: Cargando NIT's a la base de datos.")                         
                
                result = cargar_archivo(uploaded_file, "temporal.parquet") 
                if result:
                    st.write("Paso 2: Archivo cargado correctamente.")
                    st.write("Paso 3: Completando columnas del archivo NIT's")
                    completar_nits()
                    elapsed_time = time.time() - start_time
                    status.update(label=f"Proceso realizado correctamente en {elapsed_time:.2f} segundos", state="complete")
                else:
                    status.update(label="Ocurrio un error al subir el archivo", state="error")

        if st.button("Generar Recomendaciones"):
            start_time = time.time()
            with st.status("Cargando datos... por favor espera.", expanded=True) as status:
                st.write("Paso 1: Generando archivo principal sin NIT's repetidos")
                crear_base_principal()
                st.write("Paso 2: Archivo generado correctamente")
                st.write("Paso 3: Generando recomendaciones apartir del modelo ...")
                resultados = modelo_principal_sec()
                if resultados.empty:
                    status.update(label=f"Ocurrio un error en la Generacion de la recomendacion", state="error")
                else:
                    st.write("Paso 4: Creacion de archivo CSV")
                    st.download_button("Descargar CSV", resultados.to_csv(index=False, sep=";", decimal=","), "recomendaciones.csv", "text/csv")
                    elapsed_time = time.time() - start_time
                    status.update(label=f"Proceso realizado correctamente en {elapsed_time:.2f} segundos", state="complete")
                
elif opcion == "Actualizar Base Principal":
    st.title("Actualizar Base Principal")
    nuevo_archivo = st.file_uploader("Sube un nuevo archivo de base principal", type=["xlsx", "xls"])
    if nuevo_archivo:
        if st.button("Actualizar Base Principal"):
            with st.status("Cargando datos... por favor espera.", expanded=True) as status:
                start_time = time.time()
                st.write("Paso 1: Subiendo archivo principal a la base de datos...")
                result = cargar_archivo(nuevo_archivo, "BasePrincipal.parquet")
                elapsed_time = time.time() - start_time
                if result:
                    status.update(label=f"Archivo subido correctamente en {elapsed_time:.2f} segundos", state="complete")
                else:
                    status.update(label=f"Ocurrio un error en la subida del archivo", state="error")
