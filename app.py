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
import re
from datetime import datetime

st.set_page_config(
    page_title="Cuaderno de Campo · Café",
    page_icon="🌿",
    layout="wide"
)

NOMBRES_BONITOS = {
    "healthy": "Hoja sana",
    "miner": "Minador de la hoja",
    "phoma": "Phoma",
    "rust": "Roya del café",
    "cercospora": "Cercospora (mancha de hierro)",
    "red_spider": "Araña roja",
}

NOMBRES_CIENTIFICOS = {
    "healthy": "Sin patógeno detectado",
    "miner": "Leucoptera coffeella",
    "phoma": "Phoma spp.",
    "rust": "Hemileia vastatrix",
    "cercospora": "Cercospora coffeicola",
    "red_spider": "Oligonychus coffeae / Tetranychus spp.",
}

COLOR_CLASE = {
    "healthy": "#3F5C3E",
    "miner": "#B8541F",
    "phoma": "#8F4419",
    "rust": "#C1652F",
    "cercospora": "#A6742C",
    "red_spider": "#9B2C2C",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --parchment: #E7DFC8;
    --paper: #F7F2E6;
    --ink: #2B2018;
    --leaf: #3F5C3E;
    --leaf-dark: #2C4028;
    --rust: #C1652F;
    --bark: #8B7355;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background-color: var(--parchment);
    background-image:
        radial-gradient(circle at 20% 20%, rgba(139,115,85,0.06) 0%, transparent 45%),
        radial-gradient(circle at 80% 60%, rgba(63,92,62,0.05) 0%, transparent 45%);
    zoom: 0.85;
}

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: var(--bark);
    margin-bottom: 0.2rem;
}
.titulo-principal {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 600;
    font-size: 2.6rem;
    color: var(--ink);
    line-height: 1.05;
    margin-bottom: 0.1rem;
}
.subtitulo {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #5c5142;
    max-width: 46ch;
    margin-bottom: 1.6rem;
}

.pagina {
    background-color: var(--paper);
    border: 1px solid rgba(139,115,85,0.35);
    border-radius: 2px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 1px 0 rgba(139,115,85,0.15), 0 12px 24px -18px rgba(43,32,24,0.35);
    position: relative;
}
.pagina::before {
    content: "";
    position: absolute;
    top: 0; left: 28px; bottom: 0;
    width: 1px;
    background: repeating-linear-gradient(to bottom, transparent 0 6px, rgba(193,101,47,0.18) 6px 7px);
}

.cinta {
    position: absolute;
    top: -14px;
    left: 50%;
    transform: translateX(-50%) rotate(-2deg);
    width: 90px;
    height: 26px;
    background: rgba(193,101,47,0.35);
    border: 1px solid rgba(193,101,47,0.4);
    z-index: 2;
}

.etiqueta-muestra {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--bark);
    margin-top: 0.6rem;
}

.sello {
    display: flex;
    align-items: center;
    justify-content: center;
    border: 3px double var(--sello-color, var(--rust));
    color: var(--sello-color, var(--rust));
    border-radius: 50%;
    width: 128px;
    height: 128px;
    transform: rotate(-7deg);
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.sello .pct {
    font-size: 1.6rem;
    font-weight: 600;
}
.sello .etiqueta-precision {
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.75;
    margin-top: 0.1rem;
}

.entrada-titulo {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.7rem;
    margin-bottom: 0.1rem;
}
.entrada-cientifico {
    font-family: 'Fraunces', serif;
    font-style: italic;
    color: #5c5142;
    font-size: 0.95rem;
    margin-bottom: 1.1rem;
}

.seccion-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--leaf-dark);
    border-top: 1px dashed rgba(139,115,85,0.4);
    padding-top: 0.9rem;
    margin-top: 0.9rem;
}
.seccion-texto {
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--ink);
    margin-top: 0.3rem;
}

.stButton > button, .stDownloadButton > button {
    background-color: var(--leaf);
    color: var(--paper);
    border: none;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.78rem;
    padding: 0.55rem 1.4rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: var(--leaf-dark);
    color: var(--paper);
}

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.4);
    border: 1px dashed var(--bark);
    border-radius: 2px;
    padding: 0.6rem;
}
.pagina [data-testid="stImage"] img {
    max-height: 360px;
    width: auto !important;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.nota-pie {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--bark);
    margin-top: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model('modelo_cafe.h5')
    with open('labels.json') as f:
        labels = json.load(f)
    return modelo, labels

@st.cache_resource
def cargar_cliente_groq():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)

modelo, labels = cargar_modelo()
cliente_groq = cargar_cliente_groq()

def predecir(imagen_pil):
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
    nombre_bonito = NOMBRES_BONITOS.get(clase, clase)
    prompt = f"""
Eres un ingeniero agrónomo experto en el cultivo de café en Centroamérica.
Un sistema de inteligencia artificial acaba de analizar una hoja de café y detectó: {nombre_bonito}.

Genera una respuesta en español, clara y práctica para un caficultor, organizada EXACTAMENTE en estas 4 secciones,
usando esos títulos textuales al inicio de cada una (sin numerarlas):

Descripción de la enfermedad
Recomendaciones técnicas para el manejo preventivo
Buenas prácticas para el cuidado del cultivo
Acciones de seguimiento y monitoreo

Si la clase es "Hoja sana", ajusta el contenido a mantenimiento preventivo en vez de tratamiento de enfermedad.
Sé conciso pero útil, en párrafos cortos, sin usar viñetas ni asteriscos.
"""
    respuesta = cliente_groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    texto = respuesta.choices[0].message.content
    return texto.replace("**", "").replace("*", "")

def parsear_secciones(texto):
    """Divide el texto de Groq en las 4 secciones usando sus títulos como separadores."""
    titulos = [
        "Descripción de la enfermedad",
        "Recomendaciones técnicas para el manejo preventivo",
        "Buenas prácticas para el cuidado del cultivo",
        "Acciones de seguimiento y monitoreo",
    ]
    patron = "(" + "|".join(re.escape(t) for t in titulos) + ")"
    partes = re.split(patron, texto)
    secciones = {}
    actual = None
    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        if parte in titulos:
            actual = parte
        elif actual:
            secciones[actual] = secciones.get(actual, "") + parte + " "
    if not secciones:
        secciones["Recomendaciones"] = texto
    return secciones

def generar_pdf(clase, confianza, texto_groq):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    ancho, alto = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, alto - 2 * cm, "Cuaderno de Campo - Diagnostico Foliar de Cafe")

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, alto - 2.7 * cm, "Fecha: " + datetime.now().strftime('%d/%m/%Y %H:%M'))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, alto - 3.6 * cm, "Resultado: " + NOMBRES_BONITOS.get(clase, clase))
    c.drawString(2 * cm, alto - 4.2 * cm, "Confianza: " + f"{confianza:.2f}%")

    c.setFont("Helvetica", 10)
    y = alto - 5.2 * cm
    for linea in texto_groq.split("\n"):
        if y < 2 * cm:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = alto - 2 * cm
        c.drawString(2 * cm, y, linea[:110])
        y -= 0.5 * cm

    c.save()
    buffer.seek(0)
    return buffer

st.markdown('<div class="eyebrow">UTH · Inteligencia de Negocios · Proyecto de nube · Astrid Castellanos</div>', unsafe_allow_html=True)
st.markdown('<div class="titulo-principal">Cuaderno de campo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">Registro de una hoja de café para diagnóstico. '
    'Suba una foto o tómela con la cámara y el sistema anota la especie detectada, '
    'la confianza y las recomendaciones de manejo.</div>',
    unsafe_allow_html=True
)

if "resultado" not in st.session_state:
    st.session_state.resultado = None

col_izq, col_der = st.columns([1, 1.15], gap="large")

with col_izq:
    st.markdown('<div class="pagina">', unsafe_allow_html=True)
    st.markdown('<div class="cinta"></div>', unsafe_allow_html=True)
    st.markdown('<div class="etiqueta-muestra">Muestra No. ' +
                datetime.now().strftime('%y%m%d-%H%M') + '</div>', unsafe_allow_html=True)

    imagen = None
    archivo = st.file_uploader("Foto de la hoja", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if archivo is not None:
        imagen = Image.open(archivo)

    if imagen is not None:
        st.image(imagen, use_container_width=True)
        analizar = st.button("Analizar muestra", use_container_width=True)
    else:
        st.info("Aún no hay muestra registrada.")
        analizar = False

    st.markdown('</div>', unsafe_allow_html=True)

if imagen is not None and analizar:
    with st.spinner("Anotando la muestra..."):
        clase, confianza = predecir(imagen)
        st.session_state.resultado = {"clase": clase, "confianza": confianza}

with col_der:
    st.markdown('<div class="pagina">', unsafe_allow_html=True)

    if st.session_state.resultado is None:
        st.markdown('<div class="entrada-titulo">Entrada pendiente</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="seccion-texto">Cuando registre una muestra a la izquierda y presione '
            '"Analizar muestra", aquí va a aparecer el diagnóstico con las recomendaciones técnicas.</div>',
            unsafe_allow_html=True
        )
    else:
        clase = st.session_state.resultado["clase"]
        confianza = st.session_state.resultado["confianza"]
        color = COLOR_CLASE.get(clase, "#C1652F")

        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown(f'<div class="entrada-titulo">{NOMBRES_BONITOS.get(clase, clase)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="entrada-cientifico">{NOMBRES_CIENTIFICOS.get(clase, "")}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="sello" style="--sello-color:{color}">'
                f'<div><div class="pct">{confianza:.1f}%</div>'
                f'<div class="etiqueta-precision">precisión</div></div></div>',
                unsafe_allow_html=True
            )

        if cliente_groq is None:
            st.warning("Falta configurar GROQ_API_KEY en Secrets para generar las recomendaciones.")
        else:
            with st.spinner("Redactando la orientación técnica..."):
                texto_groq = generar_recomendaciones(clase)
            secciones = parsear_secciones(texto_groq)

            for titulo, contenido in secciones.items():
                st.markdown(f'<div class="seccion-eyebrow">{titulo}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="seccion-texto">{contenido.strip()}</div>', unsafe_allow_html=True)

            pdf_buffer = generar_pdf(clase, confianza, texto_groq)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "Descargar entrada en PDF",
                data=pdf_buffer,
                file_name="diagnostico_hoja_cafe.pdf",
                mime="application/pdf"
            )

    st.markdown('</div>', unsafe_allow_html=True)

nombres_clases = ", ".join(NOMBRES_BONITOS.get(c, c) for c in labels.values())
st.markdown(
    f'<div class="nota-pie">Diagnóstico generado por IA a partir de un modelo entrenado en: '
    f'{nombres_clases} · Confirme siempre con un técnico agrónomo</div>',
    unsafe_allow_html=True
)
