# Cuaderno de Campo · Detección de enfermedades en hojas de café

Proyecto de la clase de Computación en la Nube (UTH). Es una app web hecha en Streamlit que recibe la foto de una hoja de café, la pasa por un modelo de visión artificial (MobileNetV2 con transfer learning) para detectar si tiene alguna enfermedad, y usa la API de Groq para redactar recomendaciones técnicas de manejo. También deja descargar el diagnóstico en PDF.

## Qué detecta

El modelo se entrenó con 4 clases:

- Hoja sana
- Minador de la hoja (*Leucoptera coffeella*)
- Phoma (*Phoma spp.*)
- Roya del café (*Hemileia vastatrix*)

> Nota: el dataset original traía también muestras de *cercospora* y *araña roja*, pero esas carpetas venían con archivos `.zip` corruptos (no se pudieron descomprimir), así que se excluyeron del entrenamiento. Por eso `labels.json` solo tiene 4 clases y el modelo únicamente puede predecir dentro de esas 4.

## Estructura del repositorio

```
├── app.py                                    # App de Streamlit
├── modelo_cafe.h5                             # Modelo entrenado (MobileNetV2 + fine-tuning)
├── labels.json                                # Mapa índice → nombre de clase
├── requirements.txt                           # Dependencias
├── deteccion_enfermedades_cafe_corregido.ipynb # Notebook de entrenamiento (Google Colab)
└── README.md
```

## Cómo correrlo localmente

1. Clonar el repo e instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Crear el archivo `.streamlit/secrets.toml` con la llave de Groq:

   ```toml
   GROQ_API_KEY = "tu_api_key_aqui"
   ```

3. Correr la app:

   ```bash
   streamlit run app.py
   ```

## Cómo se entrenó el modelo

El notebook `deteccion_enfermedades_cafe_corregido.ipynb` (pensado para correr en Google Colab, con GPU) hace lo siguiente:

1. Sube y descomprime el `Dataset.zip` del curso.
2. Reorganiza las imágenes por clase, uniendo los zips partidos y saltando los que están corruptos.
3. Descarta `cercospora` y `red_spider` por los archivos corruptos mencionados arriba.
4. Arma los generadores de entrenamiento/validación (80/20) con aumento de datos.
5. Construye el modelo con MobileNetV2 preentrenado en ImageNet como base, congelada al inicio.
6. Entrena las capas nuevas y después hace fine-tuning descongelando las últimas 30 capas con una tasa de aprendizaje baja.
7. Evalúa el modelo y grafica precisión/pérdida.
8. Guarda `modelo_cafe.h5` y `labels.json`.

## Despliegue en la nube

La app está pensada para desplegarse en Streamlit Community Cloud: se conecta el repo de GitHub, se agrega `GROQ_API_KEY` en la sección de Secrets, y Streamlit instala `requirements.txt` y corre `app.py` automáticamente.

## Integración con Groq

Después de la predicción, la app arma un prompt con la clase detectada y le pide al modelo `llama-3.3-70b-versatile` de Groq que genere, en español, 4 secciones: descripción de la enfermedad, recomendaciones técnicas de manejo preventivo, buenas prácticas de cultivo y acciones de seguimiento. Ese texto se muestra en pantalla y también se arma en un PDF descargable con `reportlab`.

## Aviso

El diagnóstico lo genera un modelo de IA entrenado con un dataset limitado (y sin las clases de cercospora y araña roja). Es una herramienta de apoyo, no reemplaza la evaluación de un técnico agrónomo.
