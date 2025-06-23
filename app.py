import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os
import boto3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# --- Configuración de página ---
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="centered"
)

# --- Constantes ---
BUCKET_NAME = "myfakenewsdemoseast1"
MODEL_KEY = "models/fake_news_model.pkl"
VECTORIZER_KEY = "models/tfidf_vectorizer.pkl"
MAX_TEXT_LENGTH = 10000  # Límite de caracteres para el texto de entrada

# --- Función para limpieza de texto ---
def clean_text(text):
    # Eliminar URLs, menciones y hashtags
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text).strip()
    # Limitar longitud del texto
    return text[:MAX_TEXT_LENGTH]

# --- Carga segura de recursos ---
@st.cache_resource
def load_resources():
    try:
        # 1. Cargar imágenes locales
        bg_image = Image.open("assets/background_top.png")
        icon = Image.open("assets/fake_news_icon.png")
        
        # 2. Configurar cliente S3 con credenciales de secrets
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_DEFAULT_REGION"]
        )
        
        # 3. Descargar modelos
        s3.download_file(BUCKET_NAME, MODEL_KEY, 'model.pkl')
        s3.download_file(BUCKET_NAME, VECTORIZER_KEY, 'vectorizer.pkl')
        
        # 4. Cargar modelos
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        return bg_image, icon, model, vectorizer
        
    except Exception as e:
        st.error(f"""
        ## 🚨 Error loading resources
        **Details:** {str(e)}
        
        🔍 **Please check:**
        1. AWS credentials in Streamlit secrets are correct
        2. Files exist in S3 bucket:
           - s3://{BUCKET_NAME}/{MODEL_KEY}
           - s3://{BUCKET_NAME}/{VECTORIZER_KEY}
        3. Your IAM user has S3 read permissions
        4. Network connectivity to AWS
        """)
        st.stop()

# --- Interfaz principal ---
def main():
    # Cargar recursos (las credenciales se obtienen de los secrets)
    bg_image, icon, model, vectorizer = load_resources()
    
    # Header
    st.image(bg_image, use_container_width=True)
    st.divider()
    
    # Title
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image(icon, width=90)
    with col2:
        st.markdown(
            '<span style="color: #DC143C; font-size: 2.5em; font-weight: bold;">Fake News Detector AI</span>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<span style="color: #5D6D7E; font-size: 1.2em;">Enter a news text (max {MAX_TEXT_LENGTH:,} chars), and our AI will analyze its authenticity.</span>',
            unsafe_allow_html=True
        )

    # User input con límite de caracteres
    user_input = st.text_area(
        f"**Paste news text here (max {MAX_TEXT_LENGTH:,} characters):**",
        height=200,
        max_chars=MAX_TEXT_LENGTH,
        placeholder="e.g., 'Scientists discover a new energy source...'"
    )

    # Mostrar contador de caracteres
    if user_input:
        chars_remaining = MAX_TEXT_LENGTH - len(user_input)
        st.caption(f"Characters remaining: {chars_remaining:,}")

    # Analysis
    if st.button("**Analyze** 🔍", type="primary"):
        if not user_input.strip():
            st.warning("Please enter news text to analyze")
            return
            
        # Limpiar y truncar el texto si es necesario
        cleaned_text = clean_text(user_input)
        if len(user_input) > MAX_TEXT_LENGTH:
            st.warning(f"Text was truncated to {MAX_TEXT_LENGTH:,} characters for optimal analysis")
            
        try:
            text_vec = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vec)[0]
            proba = model.predict_proba(text_vec)[0] * 100

            # Results
            st.divider()
            result_col1, result_col2 = st.columns([0.7, 0.3])
            
            with result_col1:
                if prediction == 0:
                    st.success(f"### ✅ Real News\n**Confidence**: {proba[0]:.1f}%")
                else:
                    st.error(f"### ❌ Fake News\n**Confidence**: {proba[1]:.1f}%")

            with result_col2:
                prob_df = pd.DataFrame({
                    "Category": ["Real", "Fake"],
                    "Probability (%)": [proba[0], proba[1]]
                })
                st.bar_chart(prob_df.set_index("Category"))

            # WordCloud para noticias falsas
            if prediction == 1:
                st.subheader("Fake News Keywords", divider="gray")
                
                # Obtener características y pesos
                feature_names = vectorizer.get_feature_names_out()
                feature_weights = model.coef_[0]
                
                # Crear diccionario palabra:peso
                word_weights = {
                    word: abs(weight) 
                    for word, weight in zip(feature_names, feature_weights) 
                    if weight < -0.5  # Palabras con fuerte correlación a fake news
                }
                
                if word_weights:
                    wc = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='Reds',
                        max_words=50,
                        collocations=False
                    )
                    wordcloud = wc.generate_from_frequencies(word_weights)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("No significant keywords found for this fake news prediction")
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

    # Footer
    st.divider()
    st.caption("""
    <div style="text-align: center; color: #7F8C8D;">
        Made with ❤️ using Streamlit | Model by <b>Carla Domecq</b> | © 2025
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()