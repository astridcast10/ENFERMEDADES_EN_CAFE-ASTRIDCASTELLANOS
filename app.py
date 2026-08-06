import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuración de la página (Ancho completo para simular la captura)
st.set_page_config(
    page_title="AgroDetect v2.0 - Diagnóstico",
    page_icon="🌿",
    layout="wide"
)

# ----------------------------------------------------------------------
# 1. CARGA DEL MODELO DE INTELIGENCIA ARTIFICIAL
# ----------------------------------------------------------------------
CLASSES = [
    'Mancha de Hierro',
    'Roya del Cafeto',
    'Ojo de Gallo',
    'Hoja Sana'
]

@st.cache_resource
def load_model():
    # Se utiliza ResNet18 como modelo base de clasificación
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    # Si tienes el archivo de pesos entrenados 'model.pth', descomenta la línea de abajo:
    # model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Transformaciones de la imagen de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Base de conocimiento con las recomendaciones técnicas
RECOMENDACIONES = {
    'Mancha de Hierro': {
        'microorganismo': 'Cercospora coffeicola • Detectado recientemente',
        'texto': (
            "**Diferenciación a simple vista:**\n\n"
            "*Cercospora coffeicola* produce manchas circulares de 3-8 mm con centro "
            "grisáceo-necrótico y halo amarillo-anaranjado. A diferencia de la Roya, no hay "
            "pústulas en relieve ni esporas en el envés. Se confunde frecuentemente con manchas de nutrición "
            "(deficiencia de Mn), pero la distribución y el halo distintivo lo confirman."
        )
    },
    'Roya del Cafeto': {
        'microorganismo': 'Hemileia vastatrix • Detectado recientemente',
        'texto': (
            "**Diferenciación a simple vista:**\n\n"
            "Presencia de pústulas de color naranja intenso o amarillo en el envés de la hoja. "
            "Genera defoliación prematura y pérdida de rendimiento en el lote."
        )
    },
    'Ojo de Gallo': {
        'microorganismo': 'Mycena citricolor • Detectado recientemente',
        'texto': (
            "**Diferenciación a simple vista:**\n\n"
            "Manchas circulares bien definidas de color pardo o ceniza que pueden desprenderse "
            "dejando perforaciones limpias en la lámina foliar."
        )
    },
    'Hoja Sana': {
        'microorganismo': 'Ningún patógeno detectado',
        'texto': "La muestra evaluada no presenta sintomatología visible de enfermedades principales."
    }
}

# ----------------------------------------------------------------------
# 2. INTERFAZ GRÁFICA (Streamlit UI)
# ----------------------------------------------------------------------

# Encabezado
st.title("AgroDetect v2.0 - Diagnóstico Foliar")
st.caption("Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de Roya, Cercospora o Plagas.")

col1, col2 = st.columns([1, 1], gap="large")

# Columna Izquierda: Entrada y Previsualización
with col1:
    source_option = st.radio("Origen de la imagen:", ("Subir archivo", "Usar cámara"), horizontal=True)
    
    uploaded_file = None
    if source_option == "Subir archivo":
        uploaded_file = st.file_uploader("Seleccione una imagen de la hoja", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Tome una foto de la hoja")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Imagen cargada para diagnóstico", use_column_width=True)

# Columna Derecha: Resultados y Recomendaciones Técnicas
with col2:
    if uploaded_file is not None:
        # Preprocesar e inferir
        tensor_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top_prob, top_catid = torch.topk(probabilities, 1)
        clase_detectada = CLASSES[top_catid.item()]
        confianza = top_prob.item() * 100

        info = RECOMENDACIONES.get(clase_detectada, RECOMENDACIONES['Hoja Sana'])

        # Tarjeta de Diagnóstico
        st.header(f"/ {clase_detectada}")
        st.subheader(f"CONFIANZA IA: {confianza:.1f}%")
        st.caption(info['microorganismo'])

        st.divider()

        # Bloque de Recomendaciones
        st.subheader("💡 ORIENTACIÓN Y MANEJO PREVENTIVO")
        st.info(info['texto'])
    else:
        st.warning("Cargue o capture una imagen en la columna de la izquierda para generar el diagnóstico.")
