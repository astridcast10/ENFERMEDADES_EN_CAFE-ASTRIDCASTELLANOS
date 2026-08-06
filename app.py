import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from groq import Groq

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA E INTERFAZ
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AgroDetect v2.0 - Diagnóstico Foliar",
    page_icon="🌿",
    layout="wide"
)

# Estilos CSS personalizados para replicar el panel de la imagen
st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
    }
    .metric-card {
        background-color: #F3ECE6;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2D5A27;
        margin-bottom: 20px;
    }
    .rec-card {
        background-color: #F8F6F0;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2DED0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 2. CARGA DEL MODELO PYTORCH
# ----------------------------------------------------------------------
CLASSES = ['Cercospora / Mancha de Hierro', 'Roya del Cafeto', 'Miner de la Hoja', 'Phoma', 'Hoja Sana']

@st.cache_resource
def load_disease_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # Si subiste tu modelo_cafe.pth al repositorio, lo cargará:
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
# 3. INTEGRACIÓN OBLIGATORIA CON API GROQ (LLM)
# ----------------------------------------------------------------------
def generar_recomendacion_groq(enfermedad, porcentaje_confianza):
    """
    Consulta a la API de Groq para obtener recomendaciones dinámicas 
    y profesionales adaptadas al diagnóstico actual.
    """
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
    
    if not api_key:
        return """
        **1. Diferenciación a simple vista:** Lesión focal en tejido foliar con decoloración circular o irregular.
        
        **2. Manejo agronómico preventivo y correctivo:** Aumentar fertilización equilibrada (N y K). Regular sombra al 40-50%. Aplicar fungicidas a base de cobre o ingrediente activo específico según la recomendación técnica local.
        
        **3. Consulta a un técnico IHCAFE:** Mantener muestreos quincenales si la incidencia supera el 10% del lote.
        
        *(Nota: Configure GROQ_API_KEY en los Secrets de Streamlit para habilitar recomendaciones dinámicas mediante IA)*
        """

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Actúa como un experto agrónomo especialista en el cultivo de café del instituto IHCAFE.
        Se ha detectado la siguiente enfermedad en una hoja de café mediante un sistema de visión artificial:
        - Enfermedad detectada: {enfermedad}
        - Nivel de confianza del modelo: {porcentaje_confianza:.1f}%

        Genera un informe técnico estructurado exactamente con estos puntos numerados:
        1. **Diferenciación a simple vista:** Breve descripción visual distintiva de la enfermedad respecto a otras.
        2. **Manejo agronómico preventivo y correctivo:** Pasos concretos de manejo (fertilización, sombra, fungicidas/dosis).
        3. **Consulta a un técnico / Monitoreo:** Cuándo llamar a un especialista e instrucciones de seguimiento del lote.
        4. **Registro y trazabilidad:** Datos clave a documentar por el productor.

        Mantén un tono técnico, formal, claro y conciso.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al conectar con la API de Groq: {str(e)}"

# ----------------------------------------------------------------------
# 4. ESTRUCTURA VISUAL DE LA APLICACIÓN
# ----------------------------------------------------------------------
st.markdown("## **Captura de Imagen Foliar**")
st.caption("Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de Roya, Cercospora o Plagas.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    opcion_origen = st.radio("", ["📁 Subir archivo", "📷 Usar cámara"], horizontal=True)
    
    uploaded_file = None
    if "Subir" in opcion_origen:
        uploaded_file = st.file_uploader("Seleccionar imagen...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Tomar foto de la hoja")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)

with col_right:
    if uploaded_file is not None:
        # Procesar inferencia
        tensor_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        top_prob, top_catid = torch.topk(probabilities, 1)
        enfermedad_predicha = CLASSES[top_catid.item()]
        confianza_val = top_prob.item() * 100

        # Encabezado Diagnóstico (Estilo UI de la captura)
        c_diag, c_conf = st.columns([3, 1])
        with c_diag:
            st.markdown(f"### **{enfermedad_predicha}**")
            st.caption("Patógeno foliar • Detectado recientemente")
        with c_conf:
            st.markdown(f"<h2 style='text-align: right; color: #1E1E1E;'>{confianza_val:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: right; font-size: 11px; color: gray;'>CONFIANZA IA</p>", unsafe_allow_html=True)

        st.divider()

        # Llamada a Groq para generar la recomendación en tiempo real
        st.markdown("#### 💡 **ORIENTACIÓN Y MANEJO PREVENTIVO**")
        st.caption("Aquí tienes una recomendación técnica detallada generada automáticamente por Groq API:")
        
        with st.spinner("Generando análisis agronómico con la API de Groq..."):
            recomendacion_text = generar_recomendacion_groq(enfermedad_predicha, confianza_val)
        
        st.markdown(f"<div class='rec-card'>{recomendacion_text}</div>", unsafe_allow_html=True)

    else:
        st.info("👈 Por favor, suba una imagen o tome una foto en la columna izquierda para iniciar el análisis.")
