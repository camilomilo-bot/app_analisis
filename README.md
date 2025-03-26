# Aplicación de Recomendaciones con Streamlit y Azure Blob Storage


## Descripción de la Aplicación

Esta aplicación, desarrollada en Streamlit, permite la carga, procesamiento y generación de recomendaciones basadas en datos financieros y de identificación de empresas.

### Funcionalidades Principales

1. **Carga de Archivos:**

   - Permite subir archivos Excel con identificaciones (NITs o empresas).
   - Convierte los archivos a formato Parquet y los almacena en Azure Blob Storage.

2. **Procesamiento de Datos:**

   - Descarga bases de datos desde Azure Blob Storage.
   - Filtra NITs de la base principal para excluir registros repetidos.
   - Normaliza datos financieros (`Patrimonio`, `Personal`) y elimina valores no válidos.

3. **Modelo de Recomendaciones:**

   - Usa `scipy.spatial.distance.cdist` con distancia euclidiana para encontrar las mejores coincidencias entre empresas de la base principal y secundaria.
   - Filtra recomendaciones basadas en el código CIIU.
   - Permite ajustar la distancia máxima para filtrar resultados.

4. **Interfaz de Usuario en Streamlit:**

   - Dos secciones principales:
     - **"Recomendaciones"**: Permite subir un archivo y generar recomendaciones basadas en empresas similares.
     - **"Actualizar Base Principal"**: Permite cargar una nueva base principal de empresas y actualizarla en Azure Blob Storage.
   - Muestra resultados en una tabla interactiva con formato numérico.
   - Opción de descarga de resultados en formato CSV.

5. **Manejo de Errores y Conexiones:**

   - Implementa reintentos automáticos en la conexión con Azure Blob Storage para evitar fallos temporales.
   - Mensajes de error y advertencias en caso de problemas con la carga o procesamiento de datos.

## Instalación y Configuración

### Requisitos Previos

Antes de ejecutar la aplicación, asegúrate de tener instalado lo siguiente:

- Python 3.9 o superior (Recomendado Python 3.13)
- Pip
- Una cuenta de Azure con un contenedor en Blob Storage (esto es necesario si se quiere ejecutar correctamente la aplicacion)

### 1️⃣ Crear un Entorno Virtual  
Se recomienda crear un entorno virtual para aislar las dependencias del proyecto:

```bash
## En Windows
python -m venv venv
venv\Scripts\activate

## En Mac/Linux
python3 -m venv venv
source venv/bin/activate

### 2️⃣ Instalar Dependencias
Ejecuta el siguiente comando para instalar todas las dependencias necesarias:

pip install -r requirements.txt

### Configuración del Entorno

Crea un archivo `.env` en el directorio del proyecto y agrega las siguientes variables con la información de tu cuenta de Azure:
```
```ini
CONNECTION_STRING="tu_connection_string_de_azure"
CONTAINER_NAME="nombre_del_contenedor"
```
## Ejecución de la Aplicación

Ejecuta el siguiente comando en la terminal dentro del directorio del proyecto:

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501/`.


