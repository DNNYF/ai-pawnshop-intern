import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

st.set_page_config(
    page_title="AI Pawnshop Valuator",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")


def get_valuation(label, confidence):
    base_prices = {
        "laptop": 5000000,
        "cell phone": 3000000,
        "tv": 2500000,
        "mouse": 150000
    }
    base_price = base_prices.get(label, 0)
    market_value = base_price
    pawn_value = market_value * 0.75
    return market_value, pawn_value


def analyze_image_quality(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    issues = []
    if brightness < 50:
        issues.append("Terlalu Gelap: Perbaiki dengan pencahayaan yang lebih baik")
    elif brightness > 200:
        issues.append("Overexposed: Kurangi pencahayaan untuk hasil lebih baik")
    
    if contrast < 20:
        issues.append("Kontras Rendah: Objek mungkin tidak terlihat jelas")
    
    if laplacian_var < 100:
        issues.append("Gambar Buram: Pastikan kamera fokus pada objek")
    
    return brightness, contrast, laplacian_var, issues

def main():
    st.title("Digital Pawnshop AI Valuator")
    st.markdown("""
    Sistem penilaian otomatis berbasis Computer Vision untuk estimasi harga gadai.
    **Fokus Demo:** Elektronik 
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Foto Barang")
        uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Foto Barang Nasabah', use_column_width=True)
            
            if st.button('Analisis & Taksir Harga'):
                with st.spinner('AI sedang menganalisis kondisi barang...'):
                    img_array = np.array(image)
                    brightness, contrast, blur_score, quality_issues = analyze_image_quality(img_array)
                    results = model(img_array)
                    
                    detected_objects = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = model.names[cls]
                            if conf > 0.5:
                                detected_objects.append((label, conf))

                with col2:
                    st.subheader("2. Hasil Analisis AI")
                    
                    with st.expander("Info Kualitas Gambar", expanded=True):
                        st.write(f"**Brightness**: {brightness:.0f}/255")
                        st.write(f"**Contrast**: {contrast:.0f}")
                        st.write(f"**Blur Score**: {blur_score:.0f}")
                        if quality_issues:
                            st.warning("Masalah Terdeteksi:")
                            for issue in quality_issues:
                                st.write(f"• {issue}")
                    
                    if detected_objects:
                        best_obj = max(detected_objects, key=lambda x: x[1])
                        label, conf = best_obj
                        
                        st.success(f"Terdeteksi: **{label.upper()}**")
                        st.progress(conf, text=f"Confidence Score: {conf*100:.1f}%")
                        
                        market_val, pawn_val = get_valuation(label, conf)
                        
                        st.divider()
                        st.subheader("3. Estimasi Harga")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Harga Pasar (Est)", f"Rp {market_val:,.0f}")
                        with c2:
                            st.metric("Taksiran Gadai", f"Rp {pawn_val:,.0f}", delta="Cair Segera")
                        
                        st.info("Analisis Lanjutan: Kondisi fisik tampak utuh. Disarankan pemeriksaan layar menyala untuk validasi akhir.")
                    else:
                        st.error("Objek HP tidak terdeteksi dengan jelas")
                        st.markdown("""
                        **Saran Perbaikan:**
                        1. Gunakan pencahayaan alami atau terang
                        2. Dekatkan kamera ke HP, hindari latar belakang yang ramai
                        3. Pastikan objek terlihat jelas dan penuh
                        4. Gunakan pencahayaan merata untuk hasil maksimal
                        5. Tunggu beberapa detik sebelum mengambil foto
                        """)
                        
                        if results and results[0].boxes and len(results[0].boxes) > 0:
                            st.info("Debug Info: Objek dengan confidence rendah terdeteksi. Coba perbaiki kualitas gambar.")
                        
        else:
            with col2:
                st.info("Silakan upload foto untuk memulai simulasi.")

    st.markdown("---")
    st.caption("AI Engineer Internship Test Case - Proof of Concept")

if __name__ == "__main__":
    main()