import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.preprocessing import RobustScaler
from azure.storage.blob import BlobServiceClient
import io
import datetime


st.set_page_config(layout="wide")

# Número máximo de intentos de conexión
MAX_RETRIES = 3  
RETRY_DELAY = 5  # Segundos entre intentos

load_dotenv()

df_principal = None
df_secundaria = None

container_name = os.getenv("CONTAINER_NAME")

def conectar_blob_storage():
    """ Intenta conectar a Azure Blob Storage con reintentos en caso de fallo """
    connection_string = os.getenv("CONNECTION_STRING")
    
    if not connection_string:
       
        return None  # Evita crasheos y permite manejarlo en otros lugares

    for intento in range(1, MAX_RETRIES + 1):
        try:   
            return BlobServiceClient.from_connection_string(connection_string)
        except Exception as e:
            if intento < MAX_RETRIES:
                time.sleep(RETRY_DELAY)  # Esperar antes de reintentar
            else:
                st.error(f"❌ No se pudo conectar a Azure Blob Storage: {str(e)}")
                return None  # Retorna None si no se pudo conectar

# Crear conexión
blob_service_client = conectar_blob_storage()

# Si la conexión falló, se puede manejar en otro lugar
if blob_service_client is None:
    st.error("🚨 Aplicación sin acceso a Azure, Verifica la conexión. !!!!! Intenta volviendo actualizar la pagina web 😊")

def formatear_moneda(valor):
    return "${:,.2f}".format(valor)

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
        del df
        return True
    except Exception as e:
        status.update(label=f"Error al subir archivo: {e}", state="error")
        return False

def descargar_df_desde_blob(blob_name):
    """Descarga un archivo Parquet desde Azure y lo convierte en un DataFrame"""
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        # Verificar si el archivo existe antes de descargarlo
        if not blob_client.exists():
            st.error(f"❌ El archivo {blob_name} no se encontró en la base de datos.")
            return None

        stream = io.BytesIO(blob_client.download_blob().readall())
        # Convertir de Parquet a DataFrame
        df = pd.read_parquet(stream)
        return df
    except Exception as e:
        status.update(label=f"Error al descargar archivo: {e}", state="error")
        return None


def cargar_archivo(uploaded_file, filename):
    """Lee un archivo Excel, lo convierte a Parquet en memoria y lo sube a Azure Blob Storage."""
    if not uploaded_file:
        return False
    try:
        df = pd.read_excel(uploaded_file, dtype=str)
        
        if df.empty:
            st.warning("⚠️ No se puede subir un DataFrame vacío.")
            return False
        # Subir a Azure Blob Storage
        subir_df_a_blob(df, filename)  # Asegúrate de que esta función acepte un BytesIO
        return len(df)

    except Exception as e:
        status.update(label="Error: No se pudo subir un archivo a Azure", state="error")
        return 0 

def crear_base_principal():
    try:
        # Cargar archivos con Pandas en formato Parquet
        df_nits = descargar_df_desde_blob(blob_name="BaseSecundaria.parquet")
        df_datos = descargar_df_desde_blob(blob_name="BasePrincipal.parquet")

        # Eliminar valores nulos y vacíos en la columna 'ciiu_ccb'
        df_datos = df_datos[df_datos["ciiu_ccb"].notna()]  # Elimina NaN
        df_datos = df_datos[df_datos["ciiu_ccb"].str.strip() != ""]  # Elimina valores vacíos

        # Eliminar valores nulos y vacíos en la columna 'ciiu_ccb'
        df_nits = df_nits[df_nits["ciiu_ccb"].notna()]  # Elimina NaN
        df_nits = df_nits[df_nits["ciiu_ccb"].str.strip() != ""]  # Elimina valores vacíos

        # Filtrar los NITs presentes en la base grande
        df_sin_nits = df_datos[~df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]

        df_sin_nits = df_sin_nits[(df_sin_nits["Personal"] > 0) & (df_sin_nits["Patrimonio"] > 0)]  # Mantiene solo valores mayores a 0

        # Subir a Azure Blob Storage
        result = subir_df_a_blob(df_sin_nits, blob_name="BasePrincipalSNIT.parquet")

        return result  # True si se subió correctamente, False si hubo un error

    except Exception as e:
        status.update(label="Error: Creando la Base Principal sin NTS's del archivo de entrada", state="error")
        return False

def completar_nits(uploaded_file):
    # Descargar los DataFrames desde Azure Blob
    df_nits = descargar_df_desde_blob(blob_name="temporal.parquet")
    
    # Verificar que los DataFrames no estén vacíos
    if df_nits is None:
        status.update(label="Error: No se pudieron descargar los datos desde el Azure", state="error")
        return False
    
    # Verificar que df_nits tenga al menos una columna
    if df_nits.shape[1] == 0:
        status.update(label="Error: No se pudieron descargar los datos desde el Azure", state="error")
        return False

    # Obtener el nombre de la primera columna
    primera_columna = df_nits.columns[0]

    # Si el nombre de la primera columna es un número, significa que no tiene encabezado
    if str(primera_columna).isdigit():
        
        # Restaurar la primera fila como datos
        df_nits.loc[-1] = df_nits.columns  # Agregar la fila de nombres como datos
        df_nits.index = df_nits.index + 1  # Ajustar los índices
        df_nits = df_nits.sort_index()  # Reordenar correctamente

        # Asignar el nuevo nombre de columna
        df_nits.columns = ["IDENTIFICACION"]

    else:
        # Si la primera columna es "NIT", cambiar a "IDENTIFICACION"
        if primera_columna == "NIT":
            df_nits.rename(columns={"NIT": "IDENTIFICACION"}, inplace=True)
        else:
            # Si tiene otro nombre, asumir que es el identificador y renombrarlo
            df_nits.rename(columns={primera_columna: "IDENTIFICACION"}, inplace=True)


    numero_registros = len(df_nits)
    if numero_registros > 0:
        st.write(f"### Paso 1: Cargando Archivo `{uploaded_file}` con un Total de {numero_registros} registros.")
        df_datos = descargar_df_desde_blob(blob_name="BasePrincipal.parquet")

        if df_datos is None:
            status.update(label="Error: No se pudieron descargar los datos desde el Azure", state="error")
            return False
        
        num_registros = len(df_datos)
        st.write(f"### Paso 2. Cargando Base Principal con un total de {num_registros:,}".replace(",", ".") + " registros.")
        
        # Verificar que df_datos tenga la columna IDENTIFICACION
        if "IDENTIFICACION" not in df_datos.columns:
            status.update(label="Error: Ocurrio un error inesperado", state="error")
            return False
        

        # Filtrar los NITs presentes en el archivo grande
        df_filtrado = df_datos[df_datos["IDENTIFICACION"].isin(df_nits["IDENTIFICACION"])]
        #limpiar dataframe sin uno
        del df_datos
        del df_nits

        codigos_ciiu_secundaria = set(
                df_filtrado["ciiu_ccb"].dropna().unique()
            )
        #print(" codigo ciius antes de eliminar" ,codigos_ciiu_secundaria, " cantidad ", len(codigos_ciiu_secundaria))

        #df_filtrado = df_filtrado.dropna(subset=["Personal", "Patrimonio", "ciiu_ccb"]) 

        # **Filtrar eliminando los que tengan PATRIMONIO o PERSONAL en 0 o vacío**
        if "Patrimonio" in df_filtrado.columns and "Personal" in df_filtrado.columns:
            df_filtrado = df_filtrado[
                (df_filtrado["Patrimonio"].astype(float) > 0) & df_filtrado["Patrimonio"].notna() &
                (df_filtrado["Personal"].astype(float) > 0) & df_filtrado["Personal"].notna()
            ]
        st.write(f"### Paso 3. Completando la Base de NITs")
  
        # Subir el archivo filtrado a Azure Blob Storage
        return subir_df_a_blob(df=df_filtrado, blob_name="BaseSecundaria.parquet")
    else:
        status.update(label="El archivo ingresado no contiene registros para procesar... Verfica tu archivo y vuelve a intentarlo", state="error")
        return False
    
def modelo_principal_sec():
  
    base_secundaria = descargar_df_desde_blob("BaseSecundaria.parquet")
    base_principal = descargar_df_desde_blob("BasePrincipalSNIT.parquet")

    if(len(base_secundaria) > 0):

        for col in ["Patrimonio", "Personal"]:
            base_secundaria[col] = pd.to_numeric(base_secundaria[col], errors="coerce").fillna(0)
            base_principal[col] = pd.to_numeric(base_principal[col], errors="coerce").fillna(0)
        
        codigos_ciiu_secundaria = set(
                base_secundaria["ciiu_ccb"].dropna().unique()
            )
        
        #print(" codigo ciius despues de eliminar" ,codigos_ciiu_secundaria, " cantidad ", len(codigos_ciiu_secundaria))

        # Filtrar donde el código CIIU numérico de base_principal esté en los códigos de base_secundaria
        base_principal = base_principal[base_principal["ciiu_ccb"].isin(codigos_ciiu_secundaria)]
       
        scaler = RobustScaler()
        clientes_base_secundaria = scaler.fit_transform(base_secundaria[["Patrimonio", "Personal"]])
        clientes_base_principal = scaler.transform(base_principal[["Patrimonio", "Personal"]])

        #distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="euclidean")
        distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="mahalanobis")
       
        mejores_indices = np.argmin(distancias, axis=1)
        
        resultados = pd.DataFrame({
            "Identificacion": base_principal["IDENTIFICACION"],
            "EMPRESA": base_principal["EMPRESA"],
            "Patrimonio": base_principal["Patrimonio"],
            "Personal": base_principal["Personal"],
            "Codigo_CIIU": base_principal["ciiu_ccb"],
            "Distancia": distancias[np.arange(len(base_principal)), mejores_indices],

            # Mejor oferta
            "Identificacion_cliente": base_secundaria.iloc[mejores_indices]["IDENTIFICACION"].values,
            "EMPRESA_cliente": base_secundaria.iloc[mejores_indices]["EMPRESA"].values,
            "Patrimonio_cliente": base_secundaria.iloc[mejores_indices]["Patrimonio"].values,
            "Personal_cliente": base_secundaria.iloc[mejores_indices]["Personal"].values,
            "Codigo_CIIU_Cliente": base_secundaria.iloc[mejores_indices]["ciiu_ccb"].values,
            })
            
        resultados["Distancia"] = resultados["Distancia"].round(8)

        #limpiar dataframe sin uno
        del base_principal
        del base_secundaria

        return resultados
    else:
        return pd.DataFrame()

st.title("Aplicación Express")
st.write("### Sube el archivo con NITs o Identificaciónes")
uploaded_file = st.file_uploader("Sube la base con NIT o Identificación", type=["xlsx", "xls"], label_visibility="hidden")
if uploaded_file:
    if st.button("Generar Recomendaciones"):
        start_time = time.time()
        with st.status("Procesando datos... por favor espera.", expanded=True) as status:                    
            result = cargar_archivo(uploaded_file, "temporal.parquet") 
            if completar_nits(uploaded_file.name):
                    st.write("### Paso 4: Aplicando modelo usando Patrimonio y Personal...")
                    crear_base_principal()
                    resultados = modelo_principal_sec()
                    if resultados.empty:
                        status.update(label=f"Ocurrio un error en la Generacion de la recomendacion", state="error")
                        st.error("Ocurrio un error en la Generacion de la recomendacion")
                    else:
                        st.session_state.resultados = resultados
                        num_generados = len(resultados)
                        st.write(f"### Paso 5: Se generaron {num_generados:,} registros.".replace(",", "."))
                        status.update(label=f"Proceso realizado correctamente en {time.time() - start_time:.2f} segundos", state="complete")
                                    

# Verificar si ya hay resultados en session_state antes de mostrar opciones de filtrado
if "resultados" in st.session_state:
    resultados = st.session_state.resultados

    # Input para ingresar distancia máxima de filtrado
    st.markdown("###¿ Cuantos registros quieres generar ?:")
    num_registros = st.number_input(
        f"Presiona Enter para aplicar, registros disponibles {len(resultados)}",
        min_value=0, 
        step=1,
        value=0,
    )

    # Ordenar el DataFrame por distancia
    df_ordenado = resultados.sort_values(by="Distancia", ascending=True)

    # Obtener el total de registros disponibles
    total_registros = len(df_ordenado)

    if(num_registros > 0):
        if num_registros > total_registros:
            st.warning(f"El número ingresado ({num_registros}) es mayor que los registros disponibles ({total_registros}). Mostrando todos los registros disponibles.")
            df_filtrado = df_ordenado  # Mostrar todos los disponibles
        else:
            # Seleccionar los primeros 'num_registros'
            df_filtrado = df_ordenado.head(num_registros)

            # Mostrar número de registros filtrados
            num_filtrados = len(df_filtrado)

            st.write(f"### 🎯 Cantidad de registros generados {num_filtrados:,}")

        df_estilizado = df_filtrado.head(500).style.format({
        "Patrimonio": "${:,.2f}",
        "Personal": "{:,}",
        "Patrimonio_cliente" : "${:,.2f}",
        "Personal_cliente": "{:,}"
        })
        # Mostrar DataFrame con estilo visual, pero sin modificar los datos reales
        st.dataframe(df_estilizado)

        csv_data = df_filtrado.to_csv(index=False, sep=";", decimal=",", encoding="latin-1").encode("UTF-8")
        # Botón para descargar datos filtrados
        # Crear el nombre del archivo con la fecha
        nombre_archivo = f"recomendaciones-{datetime.datetime.now().strftime("%d-%m-%Y")}.csv"

        st.download_button(
            label="Descargar CSV", 
            data=csv_data,  # Pasar los datos en bytes
            file_name = nombre_archivo,
            mime="text/csv"
        )  
