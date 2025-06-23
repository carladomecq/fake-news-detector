import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os
import boto3
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

# --- Carga segura de recursos ---
@st.cache_resource
def load_resources():
    try:
        # 1. Cargar imágenes
        bg_image = Image.open("assets/background_top.png")
        icon = Image.open("assets/fake_news_icon.png")
        
        # 2. Configurar cliente S3 con manejo de errores
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name='us-east-1'
        )
        
        # 3. Descargar modelos con verificación
        if not all(key in (MODEL_KEY, VECTORIZER_KEY) for key in [MODEL_KEY, VECTORIZER_KEY]):
            raise ValueError("Claves S3 inválidas")
            
        s3.download_file(BUCKET_NAME, MODEL_KEY, '/tmp/model.pkl')
        s3.download_file(BUCKET_NAME, VECTORIZER_KEY, '/tmp/vectorizer.pkl')
        
        # 4. Cargar modelos con verificación de existencia
        if not all(os.path.exists(f) for f in ['/tmp/model.pkl', '/tmp/vectorizer.pkl']):
            raise FileNotFoundError("Modelos no descargados correctamente")
            
        model = joblib.load('/tmp/model.pkl')
        vectorizer = joblib.load('/tmp/vectorizer.pkl')
        
        return bg_image, icon, model, vectorizer
        
    except Exception as e:
        st.error(f"""
        ## 🚨 Error crítico
        **No se pudieron cargar los recursos:**  
        🔍 {str(e)}  
        📌 Verifica:  
        - Credenciales AWS en Secrets  
        - Archivos en el bucket S3  
        - Nombres de archivos locales  
        """)
        st.stop()

# --- Interfaz principal ---
def main():
    try:
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
                '<span style="color: #5D6D7E; font-size: 1.2em;">Enter a news text, and our AI will analyze its authenticity.</span>',
                unsafe_allow_html=True
            )

        # Sidebar
        with st.sidebar:
            st.header("About")
            st.markdown("""
            - **Model**: Logistic Regression (TF-IDF)
            - **Accuracy**: ~95% (English texts)
            - **Data Source**: Kaggle
            - **Storage**: AWS S3
            """)

        # User input
        user_input = st.text_area(
            "**Paste news text here:**",
            height=200,
            placeholder="e.g., 'Scientists discover a new energy source...'"
        )

        # Analysis
        if st.button("**Analyze** 🔍", type="primary"):
            if not user_input.strip():
                st.warning("Please enter news text to analyze")
                return
                
            try:
                text_vec = vectorizer.transform([user_input])
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

                # WordCloud for fake news
                if prediction == 1:
                    st.subheader("Fake News Keywords", divider="gray")
                    features = vectorizer.get_feature_names_out()
                    weights = model.coef_[0]
                    word_freq = {word: abs(weight) for word, weight in zip(features, weights)}
                    
                    fig, ax = plt.subplots()
                    WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='Reds',
                        max_words=50
                    ).generate_from_frequencies(word_freq).to_image()
                    
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

        # Footer
        st.divider()
        st.caption("""
        <div style="text-align: center; color: #7F8C8D;">
            Made with ❤️ using Streamlit | Model by <b>Carla Domecq</b> | © 2025
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()