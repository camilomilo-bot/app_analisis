import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
import gc
import datetime

st.set_page_config(layout="wide")

def reiniciar_estado():
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
    limpiar_memoria()

def limpiar_memoria():
    gc.collect()

def crear_base_principal(df_nits):
    try:
        
        df_datos = pd.read_parquet('/files/BaseCliCC.parquet')
        num_registros = len(df_datos)
        st.write(f"### Paso 2. Cargando Base Principal con un total de {num_registros:,}".replace(",", ".") + " registros.")
        nits_set = set(df_nits["IDENTIFICACION"])
        # Filtrar los NITs presentes en la base grande
        df_sin_nits = df_datos[~df_datos["IDENTIFICACION"].isin(nits_set)].copy()
        
        del df_nits, df_datos, nits_set
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
          
        df_datos = pd.read_parquet('/files/BaseCliente.parquet')
        if df_datos is None or "IDENTIFICACION" not in df_datos.columns:
            return False
        
        # Merge en lugar de isin para mayor eficiencia
        df_nits["IDENTIFICACION"] = df_nits["IDENTIFICACION"].astype(str)
        df_datos["IDENTIFICACION"] = df_datos["IDENTIFICACION"].astype(str)

        df_filtrado = df_datos.merge(df_nits, on="IDENTIFICACION", how="inner").copy()
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
        distancias = cdist(clientes_base_principal, clientes_base_secundaria, metric="euclidean")
       
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
        reiniciar_estado()
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

if "resultados" in st.session_state:
    resultados = st.session_state.resultados

    # Convertir tipos para ahorrar RAM
    resultados["Cliente"] = resultados["Cliente"].astype("int8")
    resultados["Codigo_CIIU"] = resultados["Codigo_CIIU"].astype("category")
    resultados["Descripcion_CIIU"] = resultados["Descripcion_CIIU"].astype("category")

    # Separar y ordenar
    df_cliente_1 = resultados[resultados["Cliente"] == 1].sort_values(by="Distancia")
    total_final = int(len(df_cliente_1) / 0.85)
    n_cliente_0 = total_final - len(df_cliente_1)
    df_cliente_0 = resultados[resultados["Cliente"] == 0].sort_values(by="Distancia").head(n_cliente_0)

    # Liberar resultados
    del resultados
    limpiar_memoria()

    # Concatenar clientes seleccionados
    df_filtrado = pd.concat([df_cliente_1, df_cliente_0], ignore_index=True)
    del df_cliente_1, df_cliente_0, total_final, n_cliente_0
    limpiar_memoria()

    # Datos disponibles
    grupo = df_filtrado.groupby("Cliente")
    df_rent = grupo.get_group(1)
    df_crec = grupo.get_group(0)

    total_rent = len(df_rent)
    total_crec = len(df_crec)

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
        df_rent_sel = df_rent.head(num_rent)
        df_crec_sel = df_crec.head(num_crec)
        df_filtrado_final = pd.concat([df_rent_sel, df_crec_sel]).sort_values(by="Distancia")

        del df_rent_sel, df_crec_sel
        limpiar_memoria()

        st.write(f"### 🎯 Registros generados: {len(df_filtrado_final):,}")

        df_estilizado = df_filtrado_final.head(200).style.format({
            "Patrimonio": "${:,.2f}",
            "Personal": "{:,}",
            "Patrimonio_cliente": "${:,.2f}",
            "Personal_cliente": "{:,}"
        })

        st.markdown("**📋 Mostrando solo los primeros 200 registros**")
        st.dataframe(df_estilizado)

        # CSV
        nombre_archivo = f"recomendaciones-{datetime.datetime.now().strftime('%d-%m-%Y')}.csv"
        csv_data = df_filtrado_final.to_csv(index=False, sep=";", decimal=",", encoding="latin-1").encode("UTF-8")

        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name=nombre_archivo,
            mime="text/csv"
        )

        st.subheader(f"Análisis del archivo a descargar ({len(df_filtrado_final):,} registros)")

        # Métricas resumen
        total_empresas = df_filtrado_final["Identificacion"].nunique()
        resumen = df_filtrado_final.groupby("Cliente").agg({
            "Identificacion": "nunique",
            "Patrimonio": "mean",
            "Personal": "mean"
        }).rename(index={1: "Rentabilizar", 0: "Crecimiento"})

        # Preparar métricas con valores por defecto
        empresas_rent = resumen.loc["Rentabilizar", "Identificacion"] if "Rentabilizar" in resumen.index else 0
        empresas_crec = resumen.loc["Crecimiento", "Identificacion"] if "Crecimiento" in resumen.index else 0

        porcentaje_rent = (empresas_rent / total_empresas) * 100 if total_empresas else 0
        porcentaje_crec = (empresas_crec / total_empresas) * 100 if total_empresas else 0

        patrimonio_rent = resumen.loc["Rentabilizar", "Patrimonio"] if "Rentabilizar" in resumen.index else 0
        personal_rent = resumen.loc["Rentabilizar", "Personal"] if "Rentabilizar" in resumen.index else 0

        patrimonio_crec = resumen.loc["Crecimiento", "Patrimonio"] if "Crecimiento" in resumen.index else 0
        personal_crec = resumen.loc["Crecimiento", "Personal"] if "Crecimiento" in resumen.index else 0


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

        # Top CIIU
        descripcion_ciiu = df_filtrado_final[['Codigo_CIIU', 'Descripcion_CIIU']].drop_duplicates(subset='Codigo_CIIU')
        top_ciiu = df_filtrado_final['Codigo_CIIU'].value_counts().head(5).reset_index()
        top_ciiu.columns = ['Codigo_CIIU', 'Cantidad']
        top_ciiu = top_ciiu.merge(descripcion_ciiu, how='left', on='Codigo_CIIU')

        st.subheader("Top 5 Códigos CIIU más frecuentes")
        st.dataframe(top_ciiu)

        # Limpiar
        del df_filtrado, df_filtrado_final, df_rent, df_crec, top_ciiu, descripcion_ciiu, resumen
        limpiar_memoria()
