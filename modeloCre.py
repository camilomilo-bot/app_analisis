from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

def modelo_crecimiento(base_secundaria=None, base_principal=None):
        """
            Compara una base principal de empresas con una base secundaria utilizando atributos numéricos
            para encontrar las observaciones más similares según distancia euclidiana y retorna las 5000 
            más cercanas.

            Parámetros
            ----------
            base_secundaria : pd.DataFrame
                DataFrame de referencia o comparación que contiene al menos las columnas "PATRIMONIO" y "PERSONAL".
            
            base_principal : pd.DataFrame
                DataFrame con los registros a perfilar, que debe contener las columnas:
                "IDENTIFICACION", "RAZON_SOCIAL", "TIPO_DOCUMENTO", "PATRIMONIO", "PERSONAL",
                "CIIU", "DESCRIPCION_CIIU", "CLIENTE".

            Retorna
            -------
            resultados : pd.DataFrame
                DataFrame con los 5000 registros más similares (según menor distancia euclidiana) incluyendo:
                - IDENTIFICACION
                - RAZON_SOCIAL
                - TIPO_DOCUMENTO
                - PATRIMONIO
                - PERSONAL
                - CIIU
                - DESCRIPCION_CIIU
                - CLIENTE
                - DISTANCIA (ordenada y ajustada incrementalmente desde 0.1)

            Notas
            -----
            - La distancia se calcula con atributos escalados ("PATRIMONIO" y "PERSONAL") usando `RobustScaler`
            para minimizar el impacto de outliers.
            - Se usa la métrica de distancia Euclidiana mediante `scipy.spatial.distance.cdist`.
            - La distancia original se reemplaza con una escala incremental artificial (0.1, 0.2, ..., 500.0)
            para fines de visualización o posterior interpretación.
        """
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
        
        #organizar resultados desde distanica mayor a menor
        resultados = resultados.sort_values(by="DISTANCIA", ascending=True).reset_index(drop=True)
        #tomar solo 5000 registros 
        resultados = resultados.head(5000)
        return resultados