import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import datetime
from groq import Groq

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS EXACTOS DE LA INTERFAZ
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AgroDetect v2.0 - Diagnóstico Foliar",
    page_icon="🌿",
    layout="wide"
)

# Estilos CSS personalizados para replicar el diseño crema y limpio
st.markdown("""
<style>
    /* Fondo principal y tipografía */
    .stApp {
        background-color: #F6F4EE !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Panel contenedor derecho */
    .card-container {
        background-color: #EFECE4;
        border-radius: 16px;
        padding: 24px;
        margin-top: 10px;
        border: 1px solid #E2DEC9;
    }

    /* Insignias numeradas verdes (01, 02, etc.) */
    .num-badge {
        background-color: #2D5A43;
        color: white;
        font-weight: bold;
        font-size: 13px;
        border-radius: 12px;
        padding: 4px 10px;
        display: inline-block;
        margin-right: 10px;
        vertical-align: top;
    }

    /* Títulos dentro del informe */
    .rec-item-title {
        font-weight: bold;
        color: #1A1A1A;
        font-size: 14px;
        display: inline-block;
    }

    /* Párrafos dentro del informe */
    .rec-item-text {
        color: #4A4A4A;
        font-size: 13px;
        line-height: 1.5;
        margin-left: 45px;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    /* Footer del sistema */
    .custom-footer {
        color: #8C8C8C;
        font-size: 11px;
        margin-top: 30px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 2. MODELO DE CLASIFICACIÓN (PyTorch)
# ----------------------------------------------------------------------
CLASSES = ['Cercospora / Mancha de Hierro', 'Roya del Cafeto', 'Miner de la Hoja', 'Phoma', 'Hoja Sana']

@st.cache_resource
def load_disease_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    if os.path.exists('modelo_cafe.pth'):
        model.load_state_dict(torch.load('modelo_cafe.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model

model = load_disease_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------
# 3. INTEGRACIÓN CON API DE GROQ (Formato HTML idéntico a la UI)
# ----------------------------------------------------------------------
def generar_orientacion_groq(enfermedad, porcentaje_confianza):
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    
    if not api_key:
        # Respuesta por defecto maquetada exactamente como la imagen si no hay clave API activa
        return """
        <div>
            <div>
                <span class="num-badge">01</span>
                <span class="rec-item-title">Diferenciación a simple vista</span>
                <div class="rec-item-text">Cercospora coffeicola produce manchas circulares de 3-8 mm con centro grisáceo-necrótico y halo amarillo-anaranjado. A diferencia de la Roya, no hay pústulas en relieve ni esporas en el envés. Se confunde frecuentemente con manchas de nutrición (deficiencia de Mn), pero estas carecen del halo definido y son más irregulares.</div>
            </div>
            <div>
                <span class="num-badge">02</span>
                <span class="rec-item-title">Manejo agronómico preventivo y correctivo</span>
                <div class="rec-item-text">Aumentar fertilización nitrogenada (urea foliar al 2%) y potásica. Regular sombra al 40-50% para reducir estrés hídrico. Aplicar caldo bordelés preventivo antes de lluvias intensas. En brotes activos, usar fungicidas específicos con mancozeb o clorotalonil. Evitar trabajos en campo con follaje mojado para no diseminar esporas.</div>
            </div>
            <div>
                <span class="num-badge">03</span>
                <span class="rec-item-title">Consulta a un técnico IHCAFE</span>
                <div class="rec-item-text">Consulte si las manchas aparecen en más del 10% del foliage o si persisten tras dos aplicaciones fungicidas. El técnico analizará niveles de N y K en suelo/hoja para descartar que sea un problema nutricional primario que debilite el tejido y facilite la infección.</div>
            </div>
            <div>
                <span class="num-badge">04</span>
                <span class="rec-item-title">Monitoreo y seguimiento</span>
                <div class="rec-item-text">Monitoree quincenalmente en épocas secas y calurosas (febrero-abril), cuando el estrés hídrico potencia la enfermedad. Revise hojas del tercio medio de la planta. Mejora: nuevas hojas sin manchas, recuperación del color verde intenso. Empeora: coalescencia de manchas, secado de bordes foliares.</div>
            </div>
            <div>
                <span class="num-badge">05</span>
                <span class="rec-item-title">Registro y trazabilidad</span>
                <div class="rec-item-text">Registre análisis foliar bianuales, niveles de sombra (% cobertura), tipo de sombra (Inga, Erythrina, malla), fecha de aplicaciones y condiciones climáticas previas. Documente si la parcela está en ladera expuesta al sol (mayor riesgo). Estos datos son clave para ajustar el manejo integral del cultivo.</div>
            </div>
        </div>
        """

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Actúa como un experto agrónomo del instituto IHCAFE.
        Se ha detectado la siguiente enfermedad en la hoja de café:
        - Enfermedad: {enfermedad}
        - Confianza: {porcentaje_confianza:.1f}%

        Genera exactamente 5 puntos de orientación agronómica estructurados en formato HTML utilizando este formato exacto para cada uno:

        <div>
            <span class="num-badge">NUMERO</span>
            <span class="rec-item-title">TITULO DEL PUNTO</span>
            <div class="rec-item-text">DESCRIPCION DETALLADA</div>
        </div>

        Los 5 puntos deben cubrir:
        01: Diferenciación a simple vista
        02: Manejo agronómico preventivo y correctivo
        03: Consulta a un técnico IHCAFE
        04: Monitoreo y seguimiento
        05: Registro y trazabilidad

        Retorna ÚNICAMENTE el código HTML dentro del div sin bloques de markdown ```html.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"<p style='color:red;'>Error al consultar Groq API: {str(e)}</p>"

# ----------------------------------------------------------------------
# 4. ESTRUCTURA Y LAYOUT VISUAL DE LA APLICACIÓN
# ----------------------------------------------------------------------

col1, col2 = st.columns([1.1, 1], gap="large")

# --- COLUMNA IZQUIERDA: Captura / Entrada ---
with col1:
    st.markdown("<h2 style='color: #2C2C2C; margin-bottom: 0px;'>Captura de Imagen Foliar</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6E6E6E; font-size: 12px;'>Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de Roya, Cercospora o Plagas.</p>", unsafe_allow_html=True)
    
    opcion_origen = st.radio("", ["📁 Subir archivo", "📷 Usar cámara"], horizontal=True)
    
    uploaded_file = None
    if "Subir" in opcion_origen:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

# --- COLUMNA DERECHA: Resultados ---
with col2:
    if uploaded_file is not None:
        # Inferencia del modelo
        tensor_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        top_prob, top_catid = torch.topk(probabilities, 1)
        enfermedad_predicha = CLASSES[top_catid.item()]
        confianza_val = top_prob.item() * 100

        fecha_actual = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        # Subencabezado pequeño arriba
        st.markdown(f"<div style='display: flex; justify-content: space-between; font-size: 10px; color: #8A8A8A; letter-spacing: 0.5px;'><span>ÚLTIMO DIAGNÓSTICO</span><span>{fecha_actual}</span></div>", unsafe_allow_html=True)

        # Diagnóstico y Porcentaje Gigante (Alineados como en la imagen)
        c_head1, c_head2 = st.columns([3, 1])
        with c_head1:
            st.markdown(f"<h1 style='color: #1A1A1A; margin-top: -10px; font-size: 28px;'>{enfermedad_predicha}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='color: #7A7A7A; font-size: 11px; margin-top: -15px;'>Cercospora coffeicola • Detectado recientemente</p>", unsafe_allow_html=True)
        with c_head2:
            st.markdown(f"<h1 style='text-align: right; color: #1A1A1A; font-size: 34px; margin-top: -10px;'>{confianza_val:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: right; font-size: 9px; color: #7A7A7A; margin-top: -20px; font-weight: bold;'>CONFIANZA IA</p>", unsafe_allow_html=True)

        # Panel Beige Contenedor de Recomendaciones
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("<p style='font-size: 11px; font-weight: bold; color: #2D5A43; margin-bottom: 2px;'>💡 ORIENTACIÓN Y MANEJO PREVENTIVO</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 11px; color: #6A6A6A; margin-bottom: 15px;'>Aquí tienes una recomendación técnica detallada para manejar la situación:</p>", unsafe_allow_html=True)
        
        with st.spinner("Generando orientación con Groq API..."):
            html_groq = generar_orientacion_groq(enfermedad_predicha, confianza_val)
        
        st.markdown(html_groq, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Historial reciente en la parte inferior
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: #FFFFFF; padding: 10px 15px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #EAEAEA;">
            <span style="font-size: 12px; color: #333;"><span style="color: #D9534F;">●</span> {enfermedad_predicha}</span>
            <span style="font-size: 10px; color: #AAA;">{fecha_actual}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 Por favor, carga o toma una fotografía de la hoja de café para desplegar el diagnóstico y el informe del IHCAFE.")

# Footer global
st.markdown("<div class='custom-footer'>© 2026 AGRODETECT • SOPORTE IHCAFE</div>", unsafe_allow_html=True)
