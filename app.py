import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Satellite Hazard Detector & Geo-Viewer",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Module:", ["Image Classification", "Google Earth Viewer"])

# ---------------------------------------------------------
# Load Deep Learning Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('war_debris_5class_model.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.sidebar.error("Model file not found. Place 'war_debris_5class_model.h5' in the project directory.")

CLASS_NAMES = ['Airplanes', 'Destroyed Building', 'Destroyed Vehicles', 'Military Vehicles', 'Safe Area']

# ---------------------------------------------------------
# PAGE 1: Image Classification Module
# ---------------------------------------------------------
if page == "Image Classification":
    st.title("🛸 Satellite Imagery Classification Pipeline")
    st.write("Upload a satellite image tile to analyze surface hazards and classify features.")

    uploaded_file = st.file_uploader("Choose a satellite image tile...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. فتح الصورة وتحويلها لنظام RGB
        pil_image = Image.open(uploaded_file).convert('RGB')
        st.image(pil_image, caption="Uploaded Satellite Tile", use_container_width=True)

        if st.button("Analyze & Classify Image"):
            with st.spinner("Executing Model Inference..."):
                # 2. تغيير أبعاد الصورة لـ 100x100
                img_resized = pil_image.resize((100, 100))
                
                # 3. التحويل لمصفوفة أرقام float32
                img_array = np.array(img_resized, dtype=np.float32)
                
                # 4. إضافة بُعد الـ Batch (1, 100, 100, 3)
                # ملاحظة: القسمة على 255 تتم تلقائياً داخل النموذج بفضل طبقة Rescaling
                img_batch = np.expand_dims(img_array, axis=0)

                # 5. التنبؤ
                predictions = model.predict(img_batch)[0]
                best_index = np.argmax(predictions)
                predicted_class = CLASS_NAMES[best_index]
                confidence = predictions[best_index] * 100

                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence Score: {confidence:.2f}%")

# ---------------------------------------------------------
# PAGE 2: Google Earth / Satellite Coordinate Viewer
# ---------------------------------------------------------
elif page == "Google Earth Viewer":
    st.title("🌍 Google Earth & Satellite Coordinates Inspector")
    st.write("Enter geographic coordinates to inspect target regions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        latitude = st.number_input("Latitude (خط العرض):", value=15.500700, format="%.6f")

    with col2:
        longitude = st.number_input("Longitude (خط الطول):", value=32.559900, format="%.6f")

    with col3:
        zoom_level = st.slider("Zoom Level (مستوى التقريب):", min_value=1, max_value=20, value=16)

    st.markdown("---")
    st.write(f"📍 Target Location: {latitude}, {longitude}")

    # إنشاء الخريطة
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)

    # إضافة طبقة قوقل ايرث
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='Google Satellite', overlay=False, control=True).add_to(m)
# إضافة الماركر
    folium.Marker([latitude, longitude], popup=f"Target Zone: {latitude}, {longitude}", tooltip="Selected Location", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

    # عرض الخريطة
    st_folium(m, width=1100, height=500, key="google_earth_map")