
import streamlit as st
import pandas as pd
import time
import gc
import datetime
from modeloRen import modelo_rentabilizar
from modeloCre import modelo_crecimiento

st.set_page_config(page_title="Express",layout="wide")

def reiniciar_estado():
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
    limpiar_memoria()

def limpiar_memoria():
    gc.collect()

def estandarizar_columna_nits(df):
    try:
        df_resultado = df.copy()
        if df_resultado.empty:
            status.update(label="El archivo cargado está vacío. Por favor, verifique el contenido.", state="error")
            st.error("El archivo cargado está vacío. Por favor, verifique el contenido.", state="error")
            return pd.DataFrame()
        columnas_numericas_posibles = []
        for col in df_resultado.columns:
            # Convertir a string, quitar puntos y revisar si la mayoría son números
            col_sin_na = df_resultado[col].dropna().astype(str).str.replace(".", "", regex=False)
            proporción_numerica = col_sin_na.str.isdigit().mean()
            if proporción_numerica > 0.8:  # 80% o más de los valores son numéricos
                columnas_numericas_posibles.append((col, proporción_numerica))

        if not columnas_numericas_posibles:
            status.update(label="No se encontró una columna que contenga mayoritariamente identificaciones.", state="error")
            st.error("No se encontró una columna que contenga mayoritariamente identificaciones.")
            return pd.DataFrame()
        # Elegimos la columna con mayor proporción de datos numéricos
        columna_nits = max(columnas_numericas_posibles, key=lambda x: x[1])[0]
        df_resultado = df_resultado[[columna_nits]].rename(columns={columna_nits: 'IDENTIFICACION'})
        st.write(f"### Paso 1: Leyendo archivo ingresado con {len(df_resultado):,}".replace(",", ".") + " identificaciones.")

        return df_resultado

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {str(e)}")
        return pd.DataFrame()
def modelo_principal_crecimiento(base_secundaria=None, base_principal=None):
        resultados = modelo_crecimiento(base_secundaria=base_secundaria, base_principal=base_principal)
        return resultados

def obtener_base_clientes():
    #df_clientes = pd.read_csv('../Piramodey360.csv', sep='|', encoding='latin-1', low_memory=False)
    df_clientes = pd.read_csv('/files/Piramodey360.csv', sep='|', encoding='latin-1', low_memory=False)
    df_clientes['IDENTIFICACION'] = df_clientes['IDENTIFICACION'].astype(str)
    df_clientes.columns = df_clientes.columns.str.upper()
    df_clientes['IDENTIFICACION'] = df_clientes['IDENTIFICACION'].astype(str).str.strip()
    df_clientes['CLIENTE'] = 1
    #Eliminar duplicados apartir de identificacion
    df_clientes = df_clientes.drop_duplicates(subset=['IDENTIFICACION'])
    return df_clientes

def obtener_base_cc():
    df_cc = pd.read_csv('/files/BaseCC.csv', sep='|', encoding='latin-1', low_memory=False)
    
    df_cc['IDENTIFICACION'] = df_cc['IDENTIFICACION'].astype(str)
    return df_cc

def aplicar_modelos(base_nits):
    # Obtener las bases de datos 
    df_clientes = obtener_base_clientes()
    df_cc = obtener_base_cc()
    st.write(f"### Paso 2: Cargando Base Potencial.")
    #pasar a mayusculas las columnas de base_principal
    
    df_cc.columns = df_cc.columns.str.upper()
    
    df_base_secundaria = df_clientes[df_clientes['IDENTIFICACION'].isin(base_nits['IDENTIFICACION'])].copy().reset_index(drop=True)
    if len(df_base_secundaria) == 0:
        status.update(label="No se encontraron identificaciones en la base de clientes.", state="error")
        st.error("No se encontraron identificaciones en la base de clientes.")
        return pd.DataFrame()
    else:
        st.write(f"### Paso 3: Aplicando modelos.")

        df_crecimiento = modelo_crecimiento(base_secundaria=df_base_secundaria, base_principal=df_cc)
        df_rentabilizar = modelo_rentabilizar(base_secundaria=base_nits, base_principal=df_clientes)
        

        df_resultados = unificar_bases(df_rentabilizar, df_crecimiento)
        return df_resultados

def unificar_bases(df_rentabilizar, df_crecimiento):
    #Columnas deseadas
    columnas_deseadas_rent = [
        "IDENTIFICACION",
        "RAZON_SOCIAL",
        "TIPO_DOCUMENTO",
        "PATRIMONIO",
        "PERSONAL",
        "CIIU",
        "DESCRIPCION_CIIU",
        "CLIENTE",
        "PROBABILIDAD",
    ]
    columnas_deseadas_cre = [
        "IDENTIFICACION",
        "RAZON_SOCIAL",
        "TIPO_DOCUMENTO",
        "PATRIMONIO",
        "PERSONAL",
        "CIIU",
        "DESCRIPCION_CIIU",
        "CLIENTE",
        "DISTANCIA",
    ]
    #Aplicar las columnas a cada base
    df_rentabilizar = df_rentabilizar[columnas_deseadas_rent]
    df_crecimiento = df_crecimiento[columnas_deseadas_cre]
    
    # Unir las bases de datos de rentabilizar y crecimiento
    df_unificado = pd.concat([df_rentabilizar, df_crecimiento], ignore_index=True)
    return df_unificado

st.title("Perfilador Express")
st.write("### Sube el archivo con Identificaciónes")

uploaded_file = st.file_uploader("Sube la base de Identificación", type=["xlsx", "xls"], label_visibility="hidden")
if uploaded_file:
    if st.button("Generar Recomendaciones"):
        reiniciar_estado()
        start_time = time.time()
        with st.status("Procesando datos... por favor espera.", expanded=True) as status:
            
            base_nits = pd.read_excel(uploaded_file, dtype=str)
            base_nits_correguido = estandarizar_columna_nits(base_nits)
            if len(base_nits_correguido) < 20:
                status.update(label="Error: La base no cuenta con el minimo necesario para el proceso", state="error")
                st.error("Error: La base no cuenta con el minimo necesario para el proceso")
            else:
                df_resultados = aplicar_modelos(base_nits=base_nits_correguido)
                limpiar_memoria()
                if not df_resultados.empty:
                    st.session_state.resultados = df_resultados
                    st.write(f"### Paso 4: Generando base resultado.")
                    status.update(label=f"Proceso realizado correctamente en {time.time() - start_time:.2f} segundos", state="complete")                          
                    st.success(f"Proceso realizado correctamente en {time.time() - start_time:.2f} segundos")
if "resultados" in st.session_state:
    resultados = st.session_state.resultados
    
    # Convertir tipos para ahorrar RAM
    resultados["CLIENTE"] = resultados["CLIENTE"].astype("int8")
    resultados["CIIU"] = resultados["CIIU"].astype("category")
    resultados["DESCRIPCION_CIIU"] = resultados["DESCRIPCION_CIIU"].astype("category")

    # Separar y ordenar
    df_cliente_1 = resultados[resultados["CLIENTE"] == 1].sort_values(by="PROBABILIDAD", ascending=False).reset_index(drop=True)
    total_final = int(len(df_cliente_1) / 0.80)
    n_cliente_0 = total_final - len(df_cliente_1)
    df_cliente_0 = resultados[resultados["CLIENTE"] == 0].sort_values(by="DISTANCIA", ascending=True).head(n_cliente_0).reset_index(drop=True)
    # Liberar resultados
    del resultados
    limpiar_memoria()

    df_rent = df_cliente_1
    df_crec = df_cliente_0
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
        df_filtrado_final = pd.concat([df_rent_sel, df_crec_sel])
        del df_rent_sel, df_crec_sel
        limpiar_memoria()

        st.write(f"### 🎯 Registros generados: {len(df_filtrado_final):,}")

        df_estilizado = df_filtrado_final.head(200).style.format({
            "PATRIMONIO": "${:,.2f}",
            "PERSONAL": "{:,}",
            
        })

        st.markdown("**📋 Mostrando solo los primeros 200 registros**")
        st.dataframe(df_estilizado)

        # CSV
        nombre_archivo = f"recomendaciones-{datetime.datetime.now().strftime('%d-%m-%Y')}.csv"
        #redondear a 8 decimales tanto la columna prababilidad y distancia
        df_filtrado_final["PROBABILIDAD"] = df_filtrado_final["PROBABILIDAD"].round(6)
        df_filtrado_final["DISTANCIA"] = df_filtrado_final["DISTANCIA"].round(6)
        csv_data = df_filtrado_final.to_csv(index=False, sep=";", decimal=",", encoding='utf-8')

        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name=nombre_archivo,
            mime="text/csv"
        )

        st.subheader(f"Análisis del archivo a descargar ({len(df_filtrado_final):,} registros)")

        # Métricas resumen
        total_empresas = df_filtrado_final["IDENTIFICACION"].nunique()
        resumen = df_filtrado_final.groupby("CLIENTE").agg({
            "IDENTIFICACION": "nunique",
            "PATRIMONIO": "mean",
            "PERSONAL": "mean"
        }).rename(index={1: "Rentabilizar", 0: "Crecimiento"})

        # Preparar métricas con valores por defecto
        empresas_rent = resumen.loc["Rentabilizar", "IDENTIFICACION"] if "Rentabilizar" in resumen.index else 0
        empresas_crec = resumen.loc["Crecimiento", "IDENTIFICACION"] if "Crecimiento" in resumen.index else 0

        porcentaje_rent = (empresas_rent / total_empresas) * 100 if total_empresas else 0
        porcentaje_crec = (empresas_crec / total_empresas) * 100 if total_empresas else 0

        patrimonio_rent = resumen.loc["Rentabilizar", "PATRIMONIO"] if "Rentabilizar" in resumen.index else 0
        personal_rent = resumen.loc["Rentabilizar", "PERSONAL"] if "Rentabilizar" in resumen.index else 0

        patrimonio_crec = resumen.loc["Crecimiento", "PATRIMONIO"] if "Crecimiento" in resumen.index else 0
        personal_crec = resumen.loc["Crecimiento", "PERSONAL"] if "Crecimiento" in resumen.index else 0


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
        descripcion_ciiu = df_filtrado_final[['CIIU', 'DESCRIPCION_CIIU']].drop_duplicates(subset='CIIU')
        top_ciiu = df_filtrado_final['CIIU'].value_counts().head(5).reset_index()
        top_ciiu.columns = ['CIIU', 'Cantidad']
        top_ciiu = top_ciiu.merge(descripcion_ciiu, how='left', on='CIIU')

        st.subheader("Top 5 Códigos CIIU más frecuentes")
        st.dataframe(top_ciiu)

        # Limpiar
        del df_filtrado_final, df_rent, df_crec, top_ciiu, descripcion_ciiu, resumen
        limpiar_memoria()
