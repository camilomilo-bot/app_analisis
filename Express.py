import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
import gc
import datetime
from modelo import modelo_principal_rentabilizar

st.set_page_config(layout="wide")

def reiniciar_estado():
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
    limpiar_memoria()

def limpiar_memoria():
    gc.collect()

def estandarizar_columna_nits(df):
    df_resultado = df.copy()
    
    # Verificar si el archivo tiene encabezados o si el primer registro son datos
    primera_columna = df_resultado.columns[0]
    
    # Verificar si el encabezado parece ser un NIT (dato numérico)
    es_dato = False
    try:
        # Si se puede convertir a número o si parece un NIT (solo dígitos)
        if str(primera_columna).isdigit() or str(primera_columna).replace('.', '').isdigit():
            es_dato = True
    except:
        pass
    
    if es_dato:
        # El encabezado actual es un dato, lo que significa que no hay encabezados reales
        # Guardamos la primera fila (que está en los encabezados) y restablecemos los índices
        primer_registro = pd.Series([primera_columna] + list(df_resultado.columns[1:]), 
                                   name=0)
        
        # Crear un nuevo DataFrame con el encabezado correcto
        nuevas_columnas = ['IDENTIFICACION'] + [f'Columna_{i+2}' for i in range(len(df_resultado.columns)-1)]
        df_resultado.columns = nuevas_columnas
        
        # Añadir el primer registro al principio del DataFrame
        df_resultado = pd.concat([pd.DataFrame([primer_registro.values], columns=nuevas_columnas), 
                                 df_resultado]).reset_index(drop=True)
        
    else:
        # El encabezado es un encabezado real, simplemente lo renombramos
        nuevo_nombre = {primera_columna: 'IDENTIFICACION'}
        df_resultado = df_resultado.rename(columns=nuevo_nombre)
    st.write(f"### Paso 1: Leyendo archivo ingresado {len(df_resultado)} NITs.")
    return df_resultado

def modelo_principal_crecimiento(base_secundaria=None, base_principal=None):
        base_principal = base_principal.reset_index(drop=True)
        base_secundaria = base_secundaria.reset_index(drop=True)
        scaler = RobustScaler()
        base_secundaria_scaled = scaler.fit_transform(base_secundaria[["PATRIMONIO", "PERSONAL"]])
        base_principal_scaled = scaler.transform(base_principal[["PATRIMONIO", "PERSONAL"]])
        distancias = cdist(base_principal_scaled, base_secundaria_scaled, metric="euclidean")
        mejores_indices = np.argmin(distancias, axis=1)
        
        resultados = pd.DataFrame({
            "IDENTIFICACION": base_principal["IDENTIFICACION"].reset_index(drop=True),
            "RAZON_SOCIAL": base_principal["RAZON_SOCIAL"].reset_index(drop=True),
            "TIPO_DOCUMENTO": base_principal["TIPO_DOCUMENTO"].astype(str).str.upper().reset_index(drop=True),
            "PATRIMONIO": base_principal["PATRIMONIO"].reset_index(drop=True),
            "PERSONAL": base_principal["PERSONAL"].reset_index(drop=True),
            "CIIU": base_principal["CIIU"].reset_index(drop=True),
            "DESCRIPCION_CIIU": base_principal["DESCRIPCION_CIIU"].reset_index(drop=True),
            "CLIENTE": base_principal["CLIENTE"].reset_index(drop=True),
            "DISTANCIA": distancias[np.arange(len(base_principal)), mejores_indices],
            
        })
        
        #organizar resultados desde distanica mayotr a menor
        resultados = resultados.sort_values(by="DISTANCIA", ascending=True).reset_index(drop=True)
        resultados["DISTANCIA"] = resultados["DISTANCIA"].round(6)
        resultados["DISTANCIA"] = [round(0.1 * (i + 1), 1) for i in range(len(resultados))]
        #tomar solo 5000 registros 
        resultados = resultados.head(5000)
        return resultados

def obtener_base_clientes():
    #df_clientes = pd.read_csv('../Piramodey360.csv', sep='|', encoding='latin-1', low_memory=False)
    df_clientes = pd.read_csv('/files/Piramodey360.csv', sep='|', encoding='latin-1', low_memory=False)
    df_clientes['IDENTIFICACION'] = df_clientes['IDENTIFICACION'].astype(str)
    df_clientes.columns = df_clientes.columns.str.upper()
    df_clientes['CLIENTE'] = 1
    #Eliminar duplicados apartir de identificacion
    df_clientes = df_clientes.drop_duplicates(subset=['IDENTIFICACION'])
    return df_clientes

def obtener_base_cc():
    #df_cc = pd.read_csv('../files/BaseCC.csv', sep='|', encoding='latin-1', low_memory=False)
    df_cc = pd.read_csv('/files/BaseCC.csv', sep='|', encoding='latin-1', low_memory=False)
    
    df_cc['IDENTIFICACION'] = df_cc['IDENTIFICACION'].astype(str)
    return df_cc

def aplicar_modelos(base_nits):
    # Obtener las bases de datos
    st.write(f"### Paso 2: Obteniendo bases de datos.")
    df_clientes = obtener_base_clientes()
    df_cc = obtener_base_cc()
    #pasar columna identificacion a object
    
    #pasar a mayusculas las columnas de base_principal
    
    df_cc.columns = df_cc.columns.str.upper()
    
    df_base_secundaria = df_clientes[df_clientes['IDENTIFICACION'].isin(base_nits['IDENTIFICACION'])].copy().reset_index(drop=True)
    st.write(f"### Paso 3: Aplicando modelos.")
    df_crecimiento = modelo_principal_crecimiento(base_secundaria=df_base_secundaria, base_principal=df_cc)
    df_rentabilizar = modelo_principal_rentabilizar(base_secundaria=base_nits, base_principal=df_clientes)
    #df_rentabilizar = modelo_principal_rentabilizar(base_secundaria=df_base_secundaria, base_principal=df_clientes)
    df_resultados = unificar_bases(df_rentabilizar, df_crecimiento)
    return df_resultados

def unificar_bases(df_rentabilizar, df_crecimiento):
    #Columnas deseadas
    columnas_deseadas = [
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
    df_rentabilizar = df_rentabilizar[columnas_deseadas]
    df_crecimiento = df_crecimiento[columnas_deseadas]
    
    # Unir las bases de datos de rentabilizar y crecimiento
    df_unificado = pd.concat([df_rentabilizar, df_crecimiento], ignore_index=True)
    return df_unificado

st.title("Perfilador Express")
st.write("### Sube el archivo con NITs o Identificaciónes")

uploaded_file = st.file_uploader("Sube la base con NIT o Identificación", type=["xlsx", "xls"], label_visibility="hidden")
if uploaded_file:
    if st.button("Generar Recomendaciones"):
        reiniciar_estado()
        start_time = time.time()
        with st.status("Procesando datos... por favor espera.", expanded=True) as status:
            
            base_nits = pd.read_excel(uploaded_file, dtype=str)
            
            df_resultados = aplicar_modelos(base_nits=estandarizar_columna_nits(base_nits))
            if df_resultados.empty:
                status.update(label="Error: No se encontraron registros en la base de datos.", state="error")
            else:
                limpiar_memoria()
                st.session_state.resultados = df_resultados
                st.write(f"### Paso 4: Generando base resultado.")
                status.update(label=f"Proceso realizado correctamente en {time.time() - start_time:.2f} segundos", state="complete")                          

if "resultados" in st.session_state:
    resultados = st.session_state.resultados
    
    # Convertir tipos para ahorrar RAM
    resultados["CLIENTE"] = resultados["CLIENTE"].astype("int8")
    resultados["CIIU"] = resultados["CIIU"].astype("category")
    resultados["DESCRIPCION_CIIU"] = resultados["DESCRIPCION_CIIU"].astype("category")

    # Separar y ordenar
    df_cliente_1 = resultados[resultados["CLIENTE"] == 1].head(20_000).sort_values(by="DISTANCIA")
    #df_cliente_1 = resultados[resultados["CLIENTE"] == 1].head(20_000)
    total_final = int(len(df_cliente_1) / 0.80)
    n_cliente_0 = total_final - len(df_cliente_1)
    df_cliente_0 = resultados[resultados["CLIENTE"] == 0].sort_values(by="DISTANCIA").head(n_cliente_0)
    #df_cliente_0 = resultados[resultados["CLIENTE"] == 0].head(n_cliente_0)
    # Liberar resultados
    del resultados
    limpiar_memoria()

    # Concatenar clientes seleccionados
    df_filtrado = pd.concat([df_cliente_1, df_cliente_0])
    del df_cliente_1, df_cliente_0, total_final, n_cliente_0
    limpiar_memoria()
   
    # Datos disponibles
    grupo = df_filtrado.groupby("CLIENTE")
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
        #df_filtrado_final = pd.concat([df_rent_sel, df_crec_sel]).sort_values(by="Distancia")
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
        del df_filtrado, df_filtrado_final, df_rent, df_crec, top_ciiu, descripcion_ciiu, resumen
        limpiar_memoria()
