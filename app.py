import streamlit as st
import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import time
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from openpyxl import load_workbook
import io

df_principal = None
df_secundaria = None

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cargar_archivo(uploaded_file, filename):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, dtype=str)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        df.to_parquet(file_path, index=False)
        return file_path
    return None

def crear_base_principal():

    # Cargar archivos con Dask en formato Parquet
    df_nits = dd.read_parquet("/tmp/uploads/BaseSecundaria.parquet")
    df_datos = dd.read_parquet("/tmp/BasePrincipal.parquet")
    # Filtrar los NITs presentes en la base grande
    df_sin_nits = df_datos[~df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

    # Guardar el resultado en formato Parquet
    df_sin_nits.to_parquet("/tmp/BasePrincipalSNIT.parquet", write_index=False)

    print("Archivo base_sin_nits.parquet creado exitosamente.")

def completar_nits():
    # Cargar los archivos (ahora ambos en formato Parquet)
    archivo_nits = "/tmp/uploads/BaseSecundaria.parquet"  # Archivo con solo NITs
    archivo_datos = "/tmp/BasePrincipal.parquet"  # Ahora en formato Parquet

     # Leer el archivo de NITs y mostrar las columnas disponibles
    df_nits = pd.read_parquet(archivo_nits)

    # Leer el archivo de datos
    df_datos = pd.read_parquet(archivo_datos)
    # Intentar renombrar si la columna 'NIT' existe
    if "NIT" in df_nits.columns:
        df_nits.rename(columns={"NIT": "IDENTIFICACION"}, inplace=True)

    # Filtrar los NITs presentes en el archivo grande
    df_filtrado = df_datos[df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

    # Guardar el resultado en formato Parquet
    df_filtrado.to_parquet("/tmp/uploads/BaseSecundaria.parquet", index=False)

    print("Archivo BaseSecundaria.parquet creado exitosamente.")

def modelo_principal_sec(filtrar_ciiu=True):
    base_secundaria = dd.read_parquet("/tmp/uploads/BaseSecundaria.parquet")
    base_principal = dd.read_parquet("/tmp/BasePrincipalSNIT.parquet")
    

    # 📌 Convertir nombres de columnas a minúsculas
    base_secundaria = base_secundaria.rename(columns=str.lower)
    base_principal = base_principal.rename(columns=str.lower)

    print("base secundria: ", base_secundaria.columns)
    print("base principal: ", base_principal.columns)

    for col in ["patrimonio", "personal"]:
        base_secundaria[col] = dd.to_numeric(base_secundaria[col], errors="coerce").fillna(0)
        base_principal[col] = dd.to_numeric(base_principal[col], errors="coerce").fillna(0)
    
    base_secundaria = base_secundaria.compute()
    base_principal = base_principal.compute()
    
    scaler = RobustScaler()
    clientes_base_secundaria = scaler.fit_transform(base_secundaria[["patrimonio", "personal"]])
    clientes_base_principal = scaler.transform(base_principal[["patrimonio", "personal"]])
    
    distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="euclidean")
    mejor_oferta_indices = np.argmin(distancias, axis=1)
    
    resultados = pd.DataFrame({
        "Identificacion": base_principal["identificacion"],
        "EMPRESA": base_principal["empresa"],
        "Patrimonio": base_principal["patrimonio"],
        "Personal": base_principal["personal"],
        "Codigo_CIIU": base_principal["codigo_ciiu"],
        "Distancia": distancias[np.arange(len(base_principal)), mejor_oferta_indices],
        "Codigo_CIIU_cliente": base_secundaria.iloc[mejor_oferta_indices]["codigo_ciiu"].values,
        "Identificacion_cliente": base_secundaria.iloc[mejor_oferta_indices]["identificacion"].values,
        "EMPRESA_Cliente": base_secundaria.iloc[mejor_oferta_indices]["empresa"].values,
        "Patrimonio_cliente": base_secundaria.iloc[mejor_oferta_indices]["patrimonio"].values,
        "Personal_cliente": base_secundaria.iloc[mejor_oferta_indices]["personal"].values
    })
    
    umbral = np.percentile(resultados["Distancia"], 90)
    resultados["Distancia"] = resultados["Distancia"].round(4)
    resultados = resultados[resultados["Distancia"] <= umbral]
    
    if filtrar_ciiu:
        resultados = resultados[resultados["Codigo_CIIU"] == resultados["Codigo_CIIU_cliente"]]
    
    return resultados

st.sidebar.title("Menú")
opcion = st.sidebar.radio("Seleccione una opción", ["Recomendaciones", "Actualizar Base Principal"])

if opcion == "Recomendaciones":
    st.title("Generador de Recomendaciones")
    
    uploaded_file = st.file_uploader("Sube la base secundaria", type=["xlsx", "xls"])
    if uploaded_file:
        file_path = cargar_archivo(uploaded_file, "BaseSecundaria.parquet")
        if file_path:
            completar_nits()
            st.success("Archivo cargado correctamente")
    
    filtrar_ciiu = st.checkbox("Filtrar por CIIU", value=True)
    if st.button("Generar Recomendaciones"):
        crear_base_principal()
        resultados = modelo_principal_sec(filtrar_ciiu)
        st.download_button("Descargar CSV", resultados.to_csv(index=False, sep=";", decimal=","), "recomendaciones.csv", "text/csv")
        st.dataframe(resultados)
        
elif opcion == "Actualizar Base Principal":
    st.title("Actualizar Base Principal")
    nuevo_archivo = st.file_uploader("Sube un nuevo archivo de base principal", type=["xlsx", "xls"])
    if nuevo_archivo:
        start_time = time.time()
        file_path = cargar_archivo(nuevo_archivo, "BasePrincipal.parquet")
        elapsed_time = time.time() - start_time
        if file_path:
            st.success(f"Base principal actualizada correctamente en {elapsed_time:.2f} segundos")
