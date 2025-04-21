import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from azure.storage.blob import BlobServiceClient
import io
import gc
import datetime


st.set_page_config(layout="wide")


# Número máximo de intentos de conexión
MAX_RETRIES = 3  


load_dotenv()

container_name = os.getenv("CONTAINER_NAME")

def limpiar_memoria():
    gc.collect()

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
                time.sleep(3)  # Esperar antes de reintentar
            else:
                return None  # Retorna None si no se pudo conectar

# Crear conexión
blob_service_client = conectar_blob_storage()

# Si la conexión falló, se puede manejar en otro lugar
if blob_service_client is None:
    st.error("🚨 Aplicación sin acceso, Verifica la conexión. !!!!! Intenta volviendo actualizar la pagina web 😊")


def descargar_df_desde_blob(blob_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        if not blob_client.exists():
            st.error(f"❌ El archivo {blob_name} no se encontró en Azure Blob Storage.")
            return None

        with io.BytesIO(blob_client.download_blob().readall()) as stream:
            df = pd.read_parquet(stream)
        # Conversión eficiente de columnas
        if "Cliente" in df.columns:
            df["Cliente"] = pd.to_numeric(df["Cliente"], errors="coerce").astype("int8")
        return df
    except Exception as e:
        status.update(label=f"Error al descargar archivo: {e}", state="error")
        return None


def crear_base_principal(df_nits):
    try:
        df_datos = descargar_df_desde_blob(blob_name="BaseCliCC.parquet")
        num_registros = len(df_datos)
        st.write(f"### Paso 2. Cargando Base Principal con un total de {num_registros:,}".replace(",", ".") + " registros.")
        nits_set = set(df_nits["IDENTIFICACION"])
        # Filtrar los NITs presentes en la base grande
        df_sin_nits = df_datos[~df_datos["IDENTIFICACION"].isin(nits_set)]
        
        del df_nits, df_datos
        limpiar_memoria()
        return df_sin_nits  # True si se subió correctamente, False si hubo un error

    except Exception as e:
        status.update(label=f"Error: Creando la Base Principal sin NTS's del archivo de entrada {e}", state="error")
        limpiar_memoria()
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

def completar_nits(uploaded_file):
    try:
        # Leer solo la primera columna del archivo para identificar NITs
        df_nits = pd.read_excel(uploaded_file, dtype=str, usecols=[0])

        if df_nits.empty:
            status.update(label="Error: El archivo está vacío.", state="error")
            return False

        # Identificar si hay encabezado incorrecto
        primera_columna = df_nits.columns[0]
        if str(primera_columna).isdigit():
            # Restaurar primera fila como datos si no hay encabezado
            df_nits.loc[-1] = df_nits.columns
            df_nits.index = df_nits.index + 1
            df_nits = df_nits.sort_index()
            df_nits.columns = ["IDENTIFICACION"]
        else:
            df_nits.rename(columns={primera_columna: "IDENTIFICACION"}, inplace=True)

        numero_registros = len(df_nits)
        if numero_registros == 0:
            status.update(label="El archivo no contiene registros.", state="error")
            return False

        st.write(f"### Paso 1: Cargando Archivo `{uploaded_file.name}` con {numero_registros} registros.")

        # Descargar base desde Azure
        df_datos = descargar_df_desde_blob(blob_name="BaseCliente.parquet")
        if df_datos is None or "IDENTIFICACION" not in df_datos.columns:
            return False
        
        # Merge en lugar de isin para mayor eficiencia
        df_nits["IDENTIFICACION"] = df_nits["IDENTIFICACION"].astype(str)
        df_datos["IDENTIFICACION"] = df_datos["IDENTIFICACION"].astype(str)

        df_filtrado = df_datos.merge(df_nits, on="IDENTIFICACION", how="inner")
        del df_datos, df_nits
        limpiar_memoria()

        # Validar columnas antes de filtrar
        if "Patrimonio" in df_filtrado.columns and "Personal" in df_filtrado.columns:
            df_filtrado["Patrimonio"] = pd.to_numeric(df_filtrado["Patrimonio"], errors="coerce")
            df_filtrado["Personal"] = pd.to_numeric(df_filtrado["Personal"], errors="coerce")

            df_filtrado = df_filtrado[
                (df_filtrado["Patrimonio"] > 0) &
                (df_filtrado["Personal"] > 0)
            ]

        return df_filtrado

    except Exception as e:
        status.update(label=f"Error inesperado: {str(e)}", state="error")
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

    
def modelo_principal_sec(base_secundaria=None, base_principal=None):

    st.write("### Paso 4: Aplicando modelo usando Patrimonio y Personal.")
    if(len(base_secundaria) > 0 and len(base_principal) > 0):

        for col in ["Patrimonio", "Personal"]:
            base_secundaria[col] = pd.to_numeric(base_secundaria[col], errors="coerce").fillna(0)
            base_principal[col] = pd.to_numeric(base_principal[col], errors="coerce").fillna(0)
        
        codigos_ciiu_secundaria = set(
                base_secundaria["ciiu_ccb"].unique()
            )

        # Filtrar donde el código CIIU numérico de base_principal esté en los códigos de base_secundaria
        base_principal = base_principal[base_principal["ciiu_ccb"].isin(codigos_ciiu_secundaria)]
       
        scaler = RobustScaler()
        clientes_base_secundaria = scaler.fit_transform(base_secundaria[["Patrimonio", "Personal"]])
        clientes_base_principal = scaler.transform(base_principal[["Patrimonio", "Personal"]])

        #distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="euclidean")
        distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="mahalanobis")
       
       # Crear diccionario desde los clientes que sí tienen descripción
        ciiu_dict = base_principal[base_principal["Cliente"] == 1].dropna(subset=["Descripcion_CIIU"]).drop_duplicates(subset=["ciiu_ccb"])
        ciiu_dict = dict(zip(ciiu_dict["ciiu_ccb"], ciiu_dict["Descripcion_CIIU"]))

        # Rellenar descripciones faltantes en Cliente == 0 usando el diccionario
        cond = (base_principal["Cliente"] == 0) & (base_principal["Descripcion_CIIU"].isna())
        base_principal.loc[cond, "Descripcion_CIIU"] = base_principal.loc[cond, "ciiu_ccb"].map(ciiu_dict)


        mejores_indices = np.argmin(distancias, axis=1)

        resultados = pd.DataFrame({
            "Identificacion": base_principal["IDENTIFICACION"],
            "EMPRESA": base_principal["EMPRESA"],
            "Tipo_Documento": base_principal["Tipo_Documento"].astype(str).str.upper(),
            "Patrimonio": base_principal["Patrimonio"],
            "Personal": base_principal["Personal"],
            "Codigo_CIIU": base_principal["ciiu_ccb"],
            "Descripcion_CIIU": base_principal["Descripcion_CIIU"],
            "Cliente": base_principal["Cliente"],
            "Distancia": distancias[np.arange(len(base_principal)), mejores_indices], 
            # Mejor oferta
            "Identificacion_cliente": base_secundaria.iloc[mejores_indices]["IDENTIFICACION"].to_numpy(),
            "EMPRESA_cliente": base_secundaria.iloc[mejores_indices]["EMPRESA"].to_numpy(),
            "Patrimonio_cliente": base_secundaria.iloc[mejores_indices]["Patrimonio"].to_numpy(),
            "Personal_cliente": base_secundaria.iloc[mejores_indices]["Personal"].to_numpy(),
            "Codigo_CIIU_Cliente": base_secundaria.iloc[mejores_indices]["ciiu_ccb"].to_numpy(),
            "Descripcion_CIIU_Cliente": base_secundaria.iloc[mejores_indices]["Descripcion_CIIU"].to_numpy(),
            })
            
        resultados["Distancia"] = resultados["Distancia"].round(6)

        #limpiar dataframe sin uno
        del base_principal, base_secundaria
        limpiar_memoria()
        return resultados
    else:
        return pd.DataFrame()

st.title("Perfilador Express")
st.write("### Sube el archivo con NITs o Identificaciónes")
uploaded_file = st.file_uploader("Sube la base con NIT o Identificación", type=["xlsx", "xls"], label_visibility="hidden")
if uploaded_file:
    if st.button("Generar Recomendaciones"):
        start_time = time.time()
        with st.status("Procesando datos... por favor espera.", expanded=True) as status:
            df_result = completar_nits(uploaded_file)
            del uploaded_file
            limpiar_memoria()
            if not df_result.empty:      
                    df_principal = crear_base_principal(df_result)
                    st.write(f"### Paso 3. Completando la Base de NITs.")
                    if df_principal.empty:
                        status.update(label="Error: No se pudo crear la base principal.", state="error")
                        st.error("Error: No se pudo crear la base principal.")
                    else:
                        resultados = modelo_principal_sec(df_result,df_principal)
                        if resultados.empty:
                            status.update(label=f"Ocurrio un error en la Generacion de la recomendacion", state="error")
                            st.error("Ocurrio un error en la Generacion de la recomendacion")
                        else:
                            limpiar_memoria()
                            st.session_state.resultados = resultados
                            st.write(f"### Paso 5: Generando base resultado.")
                            status.update(label=f"Proceso realizado correctamente en {time.time() - start_time:.2f} segundos", state="complete")                          

# Verificar si ya hay resultados en session_state antes de mostrar opciones de filtrado
if "resultados" in st.session_state:
    resultados = st.session_state.resultados

    # Separar y ordenar solo una vez
    df_cliente_1 = resultados[resultados["Cliente"] == 1].copy().sort_values(by="Distancia")
    
    total_final = int(len(df_cliente_1) / 0.85)

    n_cliente_0 = total_final - len(df_cliente_1)

    df_cliente_0 = resultados[resultados["Cliente"] == 0].head(n_cliente_0).copy().sort_values(by="Distancia")
    # Concatenar clientes seleccionados
    df_filtrado = pd.concat([df_cliente_1, df_cliente_0], ignore_index=True)

    # Limpiar memoria de intermedios
    del resultados
    del df_cliente_1
    del df_cliente_0
    limpiar_memoria()

    # Datos disponibles
    total_rent = (df_filtrado["Cliente"] == 1).sum()
    total_crec = (df_filtrado["Cliente"] == 0).sum()

    st.markdown("### ¿Cuántos clientes vas a gestionar?")

    col1, col2 = st.columns(2)
    with col1:
        num_rent = st.number_input(
            f"Cantidad de Rentabilizar [máx {total_rent:,}]",
            min_value=0, max_value=total_rent, value=0, step=1,
        )
    with col2:
        num_crec = st.number_input(
            f"Cantidad de Crecimiento [máx {total_crec:,}]",
            min_value=0, max_value=total_crec, value=0, step=1,
        )

    if num_rent > 0 or num_crec > 0:
        df_rent = df_filtrado[df_filtrado["Cliente"] == 1].head(num_rent)
        df_crec = df_filtrado[df_filtrado["Cliente"] == 0].head(num_crec)
        df_filtrado = pd.concat([df_rent, df_crec]).sort_values(by="Distancia")
        del df_rent, df_crec
        limpiar_memoria()

        st.write(f"### 🎯 Registros generados: {len(df_filtrado):,}")

        df_estilizado = df_filtrado.head(200).style.format({
            "Patrimonio": "${:,.2f}",
            "Personal": "{:,}",
            "Patrimonio_cliente": "${:,.2f}",
            "Personal_cliente": "{:,}"
        })

        st.markdown("**📋 Mostrando solo los primeros 200 registros**")
        st.dataframe(df_estilizado)

        # Generar nombre de archivo
        nombre_archivo = f"recomendaciones-{datetime.datetime.now().strftime('%d-%m-%Y')}.csv"

        # Convertir a CSV solo si hay registros
        csv_data = df_filtrado.to_csv(index=False, sep=";", decimal=",", encoding="latin-1").encode("UTF-8")

        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name=nombre_archivo,
            mime="text/csv"
        )

        st.subheader(f"Análisis del archivo a descargar ({len(df_filtrado):,} registros)")

        # Variables para análisis
        df_rent = df_filtrado[df_filtrado["Cliente"] == 1]
        df_crec = df_filtrado[df_filtrado["Cliente"] == 0]
        total_empresas = df_filtrado["Identificacion"].nunique()

        empresas_rent = df_rent["Identificacion"].nunique()
        porcentaje_rent = (empresas_rent / total_empresas) * 100
        patrimonio_rent = df_rent["Patrimonio"].mean()
        personal_rent = df_rent["Personal"].mean()

        empresas_crec = df_crec["Identificacion"].nunique()
        porcentaje_crec = (empresas_crec / total_empresas) * 100
        patrimonio_crec = df_crec["Patrimonio"].mean()
        personal_crec = df_crec["Personal"].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Rentabilizar")
            st.metric("Cantidad de Empresas", empresas_rent, f"{porcentaje_rent:.1f}% del total")
            st.metric("Promedio Patrimonio", f"${patrimonio_rent:,.0f}")
            st.metric("Promedio Personal", f"{personal_rent:.0f} empleados")

        with col2:
            st.markdown("### Crecimiento")
            st.metric("Cantidad de Empresas", empresas_crec, f"{porcentaje_crec:.1f}% del total")
            st.metric("Promedio Patrimonio", f"${patrimonio_crec:,.0f}")
            st.metric("Promedio Personal", f"{personal_crec:.0f} empleados")

        # TOP 5 CIIU más frecuentes
        descripcion_ciiu = df_filtrado[['Codigo_CIIU', 'Descripcion_CIIU']].drop_duplicates(subset='Codigo_CIIU')
        top_ciiu = df_filtrado['Codigo_CIIU'].value_counts().head(5).reset_index()
        top_ciiu.columns = ['Codigo_CIIU', 'Cantidad']
        top_ciiu = top_ciiu.merge(descripcion_ciiu, how='left', on='Codigo_CIIU')

        # Mostrar como tabla interactiva
        st.subheader("Top 5 Códigos CIIU más frecuentes")
        st.dataframe(top_ciiu)
