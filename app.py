import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import io
from datetime import datetime

# -------------------------------------------------------------------
# Configuración de la página
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Detección de Enfermedades en Hojas de Café",
    page_icon="🌱",
    layout="centered"
)

# -------------------------------------------------------------------
# Cargar el modelo y las clases (una sola vez, se cachea)
# -------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model('modelo_cafe.h5')
    with open('labels.json') as f:
        labels = json.load(f)  # {"0": "healthy", "1": "miner", ...}
    return modelo, labels

modelo, labels = cargar_modelo()

# -------------------------------------------------------------------
# Nombres bonitos para mostrar en pantalla (ajusten si quieren)
# -------------------------------------------------------------------
NOMBRES_BONITOS = {
    "healthy": "Hoja sana",
    "miner": "Minador de la hoja (Leaf Miner)",
    "phoma": "Phoma",
    "rust": "Roya del café (Rust)",
}

# -------------------------------------------------------------------
# Cliente de Groq (la API key se guarda en Secrets, nunca en el código)
# -------------------------------------------------------------------
@st.cache_resource
def cargar_cliente_groq():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)

cliente_groq = cargar_cliente_groq()


def predecir(imagen_pil):
    """Recibe una imagen PIL, devuelve (clase, porcentaje_confianza)."""
    img = imagen_pil.resize((224, 224)).convert('RGB')
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    pred = modelo.predict(arr)[0]
    idx = int(np.argmax(pred))
    clase = labels[str(idx)]
    confianza = float(pred[idx]) * 100
    return clase, confianza


def generar_recomendaciones(clase):
    """Le pide a Groq la descripción, recomendaciones, buenas prácticas y seguimiento."""
    nombre_bonito = NOMBRES_BONITOS.get(clase, clase)

    prompt = f"""
Eres un ingeniero agrónomo experto en el cultivo de café en Centroamérica.
Un sistema de inteligencia artificial acaba de analizar una hoja de café y detectó: {nombre_bonito}.

Genera una respuesta en español, clara y práctica para un caficultor, organizada EXACTAMENTE en estas 4 secciones con esos títulos:

1. Descripción de la enfermedad
2. Recomendaciones técnicas para el manejo preventivo
3. Buenas prácticas para el cuidado del cultivo
4. Acciones de seguimiento y monitoreo

Si la clase es "Hoja sana", ajusta el contenido a recomendaciones de mantenimiento preventivo en vez de tratamiento de enfermedad.
Sé conciso pero útil, con puntos accionables.
"""

    respuesta = cliente_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return respuesta.choices[0].message.content


def generar_pdf(clase, confianza, texto_groq):
    """Arma un PDF sencillo con el resultado y las recomendaciones."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    ancho, alto = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, alto - 2 * cm, "Reporte de Detección - Hojas de Café")

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, alto - 2.7 * cm, f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, alto - 3.6 * cm, f"Resultado: {NOMBRES_BONITOS.get(clase, clase)}")
    c.drawString(2 * cm, alto - 4.2 * cm, f"Confianza: {confianza:.2f}%")

    c.setFont("Helvetica", 10)
    y = alto - 5.2 * cm
    for linea in texto_groq.split("\n"):
        if y < 2 * cm:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = alto - 2 * cm
        c.drawString(2 * cm, y, linea[:110])  # recorta líneas muy largas
        y -= 0.5 * cm

    c.save()
    buffer.seek(0)
    return buffer


# -------------------------------------------------------------------
# Interfaz
# -------------------------------------------------------------------
st.title("🌱 Detección de Enfermedades en Hojas de Café")
st.write("Suban una foto de una hoja de café y el sistema va a detectar si tiene alguna enfermedad, "
         "además de darles recomendaciones técnicas generadas con IA.")

archivo = st.file_uploader("Subir imagen de la hoja", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    if st.button("Analizar hoja"):
        with st.spinner("Analizando la imagen..."):
            clase, confianza = predecir(imagen)

        st.success(f"Resultado: **{NOMBRES_BONITOS.get(clase, clase)}**")
        st.metric("Confianza de la predicción", f"{confianza:.2f}%")

        if cliente_groq is None:
            st.warning("No se encontró la API key de Groq (configurar GROQ_API_KEY en Secrets). "
                       "No se pueden generar las recomendaciones.")
        else:
            with st.spinner("Generando recomendaciones con IA..."):
                texto_groq = generar_recomendaciones(clase)

            st.markdown("---")
            st.markdown(texto_groq)

            # Botón para descargar el reporte en PDF
            pdf_buffer = generar_pdf(clase, confianza, texto_groq)
            st.download_button(
                label="Descargar reporte en PDF",
                data=pdf_buffer,
                file_name="reporte_hoja_cafe.pdf",
                mime="application/pdf"
            )
else:
    st.info("Esperando que suban una imagen.")
