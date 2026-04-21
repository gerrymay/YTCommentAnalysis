import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import nltk
from nltk.corpus import stopwords

# Existing imports from previous cells needed for the app
from pysentimiento import create_analyzer
from youtube_comment_downloader import YoutubeCommentDownloader

# --- Configuración Inicial y Carga de Recursos ---

# Descargar las stopwords si no están ya descargadas (se asegura que estén disponibles)
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# Cargar el analizador de sentimiento y el descargador de comentarios una sola vez
# usando st.cache_resource para optimizar el rendimiento de Streamlit.
@st.cache_resource
def load_analyzer_and_downloader():
    analyzer = create_analyzer(task="sentiment", lang="es")
    downloader = YoutubeCommentDownloader()
    return analyzer, downloader

analyzer, downloader = load_analyzer_and_downloader()

# --- Funciones de Análisis ---

def get_and_analyze_comments(video_url, max_comments_limit_str, analyzer, downloader):
    """
    Descarga comentarios de YouTube, los analiza y retorna un DataFrame con los sentimientos.
    Incluye un indicador de progreso visual.
    """
    comments_list = []
    comment_count = 0
    
    if max_comments_limit_str == "Sin límite":
        max_comments_limit = float('inf')
    else:
        max_comments_limit = int(max_comments_limit_str)

    # Indicador de progreso y texto de estado
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, comment in enumerate(downloader.get_comments(video_url)):
        if comment_count >= max_comments_limit:
            break
        
        comment_text = comment['text']
        sentiment = analyzer.predict(comment_text)
        text_sentiment = {"comentario": comment_text, "sentimiento": sentiment.output}
        comments_list.append(text_sentiment)
        comment_count += 1

        # Actualizar la barra de progreso y el texto de estado
        if max_comments_limit != float('inf'):
            progress_val = min(int((comment_count / max_comments_limit) * 100), 100)
            progress_bar.progress(progress_val)
            status_text.text(f"Analizando comentarios: {comment_count} de {max_comments_limit}...")
        else:
            # Para 'Sin límite', solo mostramos el conteo
            progress_bar.progress(min(comment_count // 100, 100)) # Simple visual cue
            status_text.text(f"Analizando comentarios: {comment_count}...")
            
    progress_bar.empty() # Limpiar la barra de progreso
    status_text.empty() # Limpiar el texto de estado

    df_comments = pd.DataFrame(comments_list)
    return df_comments

def plot_sentiment_distribution(df):
    """
    Genera un gráfico de barras de la distribución de sentimientos.
    """
    if df.empty:
        return None
    sentiment_counts = df['sentimiento'].value_counts()
    colors = {'POS': 'green', 'NEU': 'gray', 'NEG': 'red'}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, 
                palette=[colors.get(s, 'blue') for s in sentiment_counts.index], ax=ax)
    ax.set_title('Distribución de Sentimientos en los Comentarios del Video')
    ax.set_xlabel('Sentimiento')
    ax.set_ylabel('Número de Comentarios')
    plt.tight_layout()
    return fig

def plot_wordcloud(df):
    """
    Genera una nube de palabras a partir de comentarios positivos.
    """
    positive_comments = df[df['sentimiento'] == 'POS']['comentario']
    if positive_comments.empty:
        return None 
    
    all_positive_text = ' '.join(positive_comments)
    
    mask_image_path = '/content/nota-musica.png'
    mask = None
    try:
        mask = np.array(Image.open(mask_image_path))
    except FileNotFoundError:
        st.warning(f"La imagen de la máscara ('nota-musica.png') no se encontró en {mask_image_path}. "
                   "La nube de palabras se generará sin máscara.")
            
    spanish_stopwords = set(stopwords.words('spanish'))
    
    wc = WordCloud(width=800, height=400, background_color='white',
                   mask=mask, stopwords=spanish_stopwords, contour_width=3, contour_color='white',
                   collocations=False).generate(all_positive_text)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Nube de Palabras de Comentarios Positivos')
    plt.tight_layout()
    return fig

# --- Estructura de la Aplicación Streamlit ---

st.set_page_config(layout="wide", page_title="Análisis de Comentarios de YouTube") 

st.title("Análisis de Sentimientos de Comentarios de YouTube")
st.markdown("Una herramienta para entender la opinión pública de tus videos.")
st.markdown("---")

# Barra lateral para las entradas del usuario
with st.sidebar:
    st.header("Configuración del Análisis")
    video_url = st.text_input("Ingresa la URL del video de YouTube:", 
                              "https://www.youtube.com/watch?v=PX-iemNhRMI&list=RDPX-iemNhRMI&start_radio=1") # Valor por defecto
    
    max_comments_options = ["100", "500", "1000", "Sin límite"]
    max_comments_selection = st.selectbox("Máximo de comentarios a analizar:", max_comments_options, index=1) # Por defecto: 500
    
    st.markdown("--- Tendrás una vista previa de la app cuando la ejecutes en un navegador --- ")
    analyze_button = st.button("Ejecutar Análisis", use_container_width=True, type="primary")

# Lógica principal de la aplicación
if analyze_button:
    if not video_url:
        st.error("¡Advertencia! Por favor, ingresa una URL de YouTube válida para continuar.")
    else:
        st.subheader("Resultados del Análisis")
        st.markdown("---")
        
        # Ayuda visual durante la descarga y el análisis de comentarios
        with st.spinner("Descargando y analizando comentarios... Esto puede tardar unos minutos, por favor espera."):
            df_comments = get_and_analyze_comments(video_url, max_comments_selection, analyzer, downloader)
        
        if df_comments.empty:
            st.warning("No se encontraron comentarios o no se pudieron analizar para la URL proporcionada. "
                       "Intenta con otra URL o verifica tu conexión a internet.")
        else:
            st.success(f"Análisis completado para {len(df_comments)} comentarios.")
            st.markdown("--- Análisis de Sentimientos --- ")
            
            # Diseño de dashboard con dos columnas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribución de Sentimientos")
                sentiment_fig = plot_sentiment_distribution(df_comments)
                if sentiment_fig:
                    st.pyplot(sentiment_fig)
                else:
                    st.info("No se pudo generar el gráfico de distribución de sentimientos.")
            
            with col2:
                st.subheader("Nube de Palabras de Comentarios Positivos")
                wordcloud_fig = plot_wordcloud(df_comments)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("No hay suficientes comentarios positivos para generar una nube de palabras o la máscara no se encontró.")
            
            st.markdown("--- Datos en Crudo ---")
            st.subheader("Tabla de Comentarios Analizados")
            st.dataframe(df_comments) # Mostrar el DataFrame completo
else:
    st.info("Ingresa una URL de YouTube en la barra lateral izquierda y haz clic en 'Ejecutar Análisis' para comenzar a visualizar los sentimientos de los comentarios.")