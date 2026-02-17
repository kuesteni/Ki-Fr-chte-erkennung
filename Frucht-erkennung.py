import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# ==========================
# Seitenkonfiguration
# ==========================
st.set_page_config(page_title="Obst KI", page_icon="🍎")

st.title("🍎🍌🍊 Obst Erkennungs KI")
st.write("Lade ein Bild hoch und die KI erkennt Apfel, Banane oder Orange.")

# ==========================
# Modell & Labels laden
# ==========================
@st.cache_resource
def load_keras_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_keras_model()

# ==========================
# Datei Upload
# ==========================
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # ==========================
    # Bild Preprocessing (wie dein Code)
    # ==========================
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ==========================
    # Vorhersage
    # ==========================
    with st.spinner("🔍 KI analysiert das Bild..."):
        prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()  # Entfernt "0 " am Anfang
    confidence_score = float(prediction[0][index])

    # ==========================
    # Ergebnis anzeigen
    # ==========================
    st.subheader("🎯 Ergebnis")

    st.success(f"**Erkannte Klasse:** {class_name}")
    st.info(f"**Sicherheit:** {confidence_score * 100:.2f}%")

    st.subheader("📊 Wahrscheinlichkeiten aller Klassen")

    for i, prob in enumerate(prediction[0]):
        label = class_names[i][2:].strip()
        st.write(f"{label}: {prob * 100:.2f}%")
        st.progress(float(prob))
