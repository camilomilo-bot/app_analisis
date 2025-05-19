import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

top_features = [
    'PRODUCTO_HOMOLOGADO', 'CFM_FIJO_TOTAL', 'SERVICIOS_FIJO_TOTAL', 'CFM_MOVIL_TOTAL',
    'PATRIMONIO', 'PERSONAL', 'ANTIGUEDAD_EMPRESA'
]

def preparar_datos(df_nits, df_base):
    """
    Cruza los NITs del DataFrame df_nits con la base completa df_base para extraer 
    la información correspondiente a esos NITs.

    Parámetros:
    - df_nits: DataFrame con columna 'IDENTIFICACION' (solo NITs).
    - df_base: DataFrame con la columna 'IDENTIFICACION' y más datos.

    Retorna:
    - DataFrame con las filas completas de df_base correspondientes a los NITs dados.
    """

    # Asegurar que las columnas estén en mayúsculas
    df_nits.columns = df_nits.columns.str.upper()
    df_base.columns = df_base.columns.str.upper()

    # Filtrar base por los NITs
    df_resultado = df_base[df_base['IDENTIFICACION'].isin(df_nits['IDENTIFICACION'])].copy()
    #convertir a csv
    #df_resultado.to_csv("df_nits_ingresados.csv", index=False, sep=";", decimal=",")
    
    return df_resultado

def agregar_variables(df_resultado):
    df_resultado = df_resultado[top_features].copy()
    
    #creamos columna target = 1
    df_resultado['TARGET'] = 1
    #Modificamos columna producto_homologado a producto_a_buscar
    df_resultado['PRODUCTO_HOMOLOGADO'] = "PRODUCTO_A_BUSCAR"
    
    return df_resultado

def muestreo_estratificado(df_resultado, df_base):
    """
    Realiza un muestreo estratificado de la base principal según la columna 'SECTOR',
    que se reconstruye a partir de variables one-hot codificadas.
    
    El tamaño del muestreo es 3 veces el número de registros de df_resultado.

    Parámetros:
    - df_resultado: DataFrame de entrada con los NITs cruzados (filtrado).
    - df_base: DataFrame completo con columnas one-hot de sector.

    Retorna:
    - DataFrame con la muestra estratificada.
    """
    df_base = df_base[~df_base['IDENTIFICACION'].isin(df_resultado['IDENTIFICACION'])].copy()
    # Tamaño deseado: 2 veces el tamaño del df_resultado
    n_total = 3 * len(df_resultado)
    #Eliminar identificaciones de df_base de los que estan en df_resultado
    
    # Calcular proporciones
    proporciones = df_base['SECTOR'].value_counts(normalize=True)
    print("Proporciones de cada clase:\n", proporciones)
    # Calcular cuántas muestras por clase
    muestras_por_clase = (proporciones * n_total).round().astype(int)
    print("Muestras por clase:\n", muestras_por_clase)
    # Hacer el muestreo estratificado
    df_muestra = (
    df_base.groupby('SECTOR', group_keys=False)
           .apply(lambda x: x.sample(
               n=min(muestras_por_clase.get(x.name, 0), len(x)),
               random_state=42
           ))
    )
    #df_muestra.to_csv("df_muestra.csv",sep=";" ,index=False, decimal=",")

    
    df_muestra = df_muestra[top_features]
    
    #Creamos columna target = 0
    df_muestra['TARGET'] = 0
    #Modificamos columna producto_homologado a no_producto_a_buscar
    df_muestra['PRODUCTO_HOMOLOGADO'] = "NO_PRODUCTO_A_BUSCAR"
    
    return df_muestra.reset_index(drop=True)


def preparar_datos_modelo(df_targets, df_targetn, test_size=0.2, random_state=42):
    """
    Une dos dataframes, separa X (features) e y (target), y los divide en entrenamiento y prueba.
    
    Retorna:
    - X_train, X_test, y_train, y_test
    """
    df_modelo = pd.concat([df_targets, df_targetn], ignore_index=True)

    # Separar características y etiqueta
    X = df_modelo.drop(columns=['PRODUCTO_HOMOLOGADO', 'TARGET'])
    y = df_modelo['TARGET']
    # Dividir los datos
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def entrenar_modelo_basico(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo RandomForest básico y muestra métricas de desempeño.
    
    Retorna:
    - modelo entrenado
    """
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("✅ Modelo básico entrenado\n")
    print("🔍 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\n📋 Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print(f"\n🎯 Accuracy entrenamiento: {accuracy_score(y_train, y_train_pred):.2f}")
    print(f"🧪 Accuracy test/validación: {accuracy_score(y_test, y_pred):.2f}")

    return model


def ajustar_random_forest(X_train, y_train, X_test, y_test, cv=5, n_estimators=100):
    """
    Realiza búsqueda de hiperparámetros con GridSearchCV sobre RandomForest.
    
    Retorna:
    - mejor modelo encontrado
    """
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    # Obtener matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Crear un DataFrame con etiquetas
    cm_df = pd.DataFrame(cm,
                        index=["Actual 0", "Actual 1"],
                        columns=["Predicho 0", "Predicho 1"])

    print("🔍 Matriz de confusión:\n")
    print(cm_df)
    print("\n📋 Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print(f"\n🎯 Accuracy entrenamiento: {accuracy_score(y_train, y_train_pred):.2f}")
    print(f"🧪 Accuracy test/validación: {accuracy_score(y_test, y_pred):.2f}")

    return best_model

def aplicar_modelo(df_nits, df_base, modelo):
    """
    Aplica un modelo entrenado al subconjunto de df_base que no está en df_nits,
    usando únicamente las columnas usadas en entrenamiento, pero devolviendo todas las columnas originales.

    Parámetros:
    - df_nits: DataFrame con columna 'IDENTIFICACION' (NITs ya utilizados).
    - df_base: DataFrame completo con todas las empresas.
    - modelo: Modelo entrenado.
    - feature_cols: Lista de columnas con las que fue entrenado el modelo.

    Retorna:
    - DataFrame con todas las columnas originales + predicción y probabilidad.
    """

    df_nits = df_nits.copy()
    df_base = df_base.copy()
    df_nits.columns = df_nits.columns.str.upper()
    df_base.columns = df_base.columns.str.upper()

    # Filtrar registros no presentes en df_nits
    df_restante = df_base[~df_base['IDENTIFICACION'].isin(df_nits['IDENTIFICACION'])].copy()
    #df_restante = df_base
    # Validar columnas de entrada para predicción
    X_restante = df_restante[top_features].copy()
    # Eliminar la columna 'PRODUCTO_HOMOLOGADO' si existe
    if 'PRODUCTO_HOMOLOGADO' in X_restante.columns:
        X_restante.drop(columns=['PRODUCTO_HOMOLOGADO'], inplace=True)
    # Realizar predicciones
    df_restante['PROBABILIDAD'] = modelo.predict_proba(X_restante)[:, 1]

    return df_restante

def modelo_rentabilizar(base_secundaria=None, base_principal=None):

    df_base_secundaria = preparar_datos(base_secundaria, base_principal)
    df_muestra_base_principal = muestreo_estratificado(df_resultado=df_base_secundaria, df_base=base_principal)

    df_base_secundaria = agregar_variables(df_base_secundaria)

    X_train, X_test, y_train, y_test = preparar_datos_modelo(df_base_secundaria, df_muestra_base_principal)
    modelo_ajustado = ajustar_random_forest(X_train, y_train, X_test, y_test)
    
    df_resultado = aplicar_modelo(df_nits=base_secundaria, df_base=base_principal, modelo=modelo_ajustado)

    #Cuantos datos tiene probabilidad mayor a 0.5
    print("Cantidad de datos con probabilidad mayor a 0.5: ", len(df_resultado[df_resultado['PROBABILIDAD'] >= 0.5]))
    #Cuantos datos tiene probabilidad mayor a 0.7
    print("Cantidad de datos con probabilidad mayor a 0.7: ", len(df_resultado[df_resultado['PROBABILIDAD'] >= 0.7]))


    #df_resultado[df_resultado['PROBABILIDAD'] > 0.5].to_csv("df_resultado_probabilidad.csv", index=False, sep=";", decimal=",")
    #df_resultado = df_resultado.sort_values(by='PROBABILIDAD', ascending=False).head(8000)
    df_resultado = df_resultado[df_resultado['PROBABILIDAD'] >= 0.7].reset_index(drop=True)

    return df_resultado

def modelo_principal_rentabilizar_filtro(base_secundaria=None, base_principal=None):
    df_base_secundaria = agregar_variables(base_secundaria)
    df_muestra_base_principal = muestreo_estratificado(df_resultado=df_base_secundaria, df_base=base_principal)
    X_train, X_test, y_train, y_test = preparar_datos_modelo(df_base_secundaria, df_muestra_base_principal)
    modelo_ajustado = ajustar_random_forest(X_train, y_train, X_test, y_test)
    df_resultado = aplicar_modelo(df_nits=base_secundaria, df_base=base_principal, modelo=modelo_ajustado)
    df_resultado = df_resultado.sort_values(by='PROBABILIDAD', ascending=False).head(8000)
    df_resultado = df_resultado.rename(columns={"PROBABILIDAD": "DISTANCIA"})
    df_resultado["DISTANCIA"] = [round(0.1 * (i + 1), 1) for i in range(len(df_resultado))]

    return df_resultado
