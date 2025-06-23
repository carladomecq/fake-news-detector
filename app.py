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

# --- Carga de recursos ---
@st.cache_resource
def load_resources():
    # Cargar imágenes
    bg_image = Image.open("assets/background_top.png")
    icon = Image.open("assets/fake_news_icon.png")
    
    # Cargar modelo (versión con S3)
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_KEY"),
            region_name='us-east-1'
        )
        
        s3.download_file('myfakenewsdemoseast1', 'models/fake_news_model.pkl', 'fake_news_model.pkl')
        s3.download_file('myfakenewsdemoseast1', 'models/tfidf_vectorizer.pkl', 'tfidf_vectorizer.pkl')
        
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return bg_image, icon, model, vectorizer
        
    except Exception as e:
        st.error(f"Error cargando recursos: {e}")
        return None, None, None, None

bg_image, icon, model, vectorizer = load_resources()

if model is None:
    st.stop()  # Detener la app si no hay modelos

# --- Interfaz de usuario ---
# Fondo decorativo
st.image(bg_image, use_container_width=True)
st.divider()

# Título con ícono
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image(icon, width=90)
with col2:
    st.markdown(
        '<span style="color: #DC143C; font-size: 2.5em; font-weight: bold;">Fake News Detector AI</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span style="color: #5D6D7E; font-size: 1.2em; text-align: left;">Enter a news text, and our AI will analyze its authenticity.</span>',
        unsafe_allow_html=True
    )

# Barra lateral
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Model**: Logistic Regression (TF-IDF)
    - **Accuracy**: ~95% (on English texts)
    - **Data Source**: Kaggle Fake & Real News Dataset
    """)

# Input de usuario
user_input = st.text_area(
    "**Paste news text here:**",
    height=200,
    placeholder="e.g., 'Scientists discover a new energy source...'"
)

# Procesamiento y resultados
if st.button("**Analyze** 🔍", type="primary"):
    if user_input:
        # Predicción
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec)[0] * 100

        # Resultados
        st.divider()
        if prediction == 0:
            st.success(f"""
            ### ✅ Real News
            **Confidence**: {proba[0]:.1f}%
            """)
        else:
            st.error(f"""
            ### ❌ Fake News
            **Confidence**: {proba[1]:.1f}%
            """)

        # Gráfico de probabilidades
        st.subheader("Prediction Probability", divider="gray")
        prob_df = pd.DataFrame({
            "Category": ["Real", "Fake"],
            "Probability (%)": [proba[0], proba[1]]
        })
        st.bar_chart(prob_df, x="Category", y="Probability (%)")

        # WordCloud para noticias falsas
        if prediction == 1:
            feature_names = vectorizer.get_feature_names_out()
            fake_coef = model.coef_[0]
            word_weights = {word: abs(weight) for word, weight in zip(feature_names, fake_coef)}
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='Reds',
                max_words=50
            ).generate_from_frequencies(word_weights)
            
            st.subheader("Key Words in Fake News", divider="gray")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

# Pie de página
st.divider()
st.caption("""
<div style="text-align: center; color: #7F8C8D;">
    Made with ❤️ using <b>Streamlit</b> | Model trained with <b>scikit-learn</b>, by <b>Carla Domecq</b> | © 2025 
</div> 
""", unsafe_allow_html=True)