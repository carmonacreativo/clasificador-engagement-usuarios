# ============================================
# APP.PY — Clasificador de Engagement
# ============================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuración de página ──
st.set_page_config(
    page_title="Clasificador de Usuarios",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS ──
st.markdown("""
<style>
    .main { background-color: #0F0F1A; }
    .stApp { background-color: #0F0F1A; }
    h1, h2, h3, p, label { color: white !important; }
    .metric-card {
        background: #1A1A2E;
        border: 1px solid #00C9A7;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .resultado-activo {
        background: linear-gradient(135deg, #00C9A7, #007A65);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .resultado-pasivo {
        background: linear-gradient(135deg, #FFB347, #CC7A00);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .resultado-nuevo {
        background: linear-gradient(135deg, #4A90D9, #1A5499);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .stSlider > div { color: white; }
    .stSelectbox > div { color: white; }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ──
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_clasificador.pkl')
    le_app = joblib.load('encoder_app.pkl')
    le_target = joblib.load('encoder_target.pkl')
    return modelo, le_app, le_target

modelo, le_app, le_target = cargar_modelo()

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🎯 Clasificador de Engagement")
    st.markdown("---")
    pagina = st.radio(
        "Navegación",
        ["🏠 Predictor", "📊 Dashboard", "ℹ️ Acerca del modelo"]
    )
    st.markdown("---")
    st.markdown("**Modelo:** Random Forest")
    st.markdown("**Accuracy:** 74.5%")
    st.markdown("**Dataset:** Kaggle — 1,000 usuarios")

print("✅ App configurada")

# ── PÁGINA 1: PREDICTOR ──
if pagina == "🏠 Predictor":

    st.markdown("# 🎯 Clasificador de Engagement de Usuarios")
    st.markdown("Ingresa los datos del usuario y el modelo predice su nivel de actividad.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Datos del usuario")

        minutos = st.slider(
            "⏱️ Minutos diarios en la app",
            min_value=5, max_value=500, value=120, step=5
        )

        likes = st.slider(
            "❤️ Likes recibidos por día",
            min_value=0, max_value=200, value=50, step=5
        )

        app_opciones = list(le_app.classes_)
        app_seleccionada = st.selectbox("📱 Plataforma utilizada", app_opciones)

        predecir = st.button("🔍 Clasificar Usuario", use_container_width=True)

    with col2:
        st.markdown("### 🎯 Resultado")

        if predecir:
            # Predicción
            app_encoded = le_app.transform([app_seleccionada])[0]
            datos = pd.DataFrame(
                [[minutos, likes, app_encoded]],
                columns=['Daily_Minutes_Spent', 'Likes_Per_Day', 'App_encoded']
            )
            pred = modelo.predict(datos)[0]
            probs = modelo.predict_proba(datos)[0]
            tipo = le_target.inverse_transform([pred])[0]

            # Mostrar resultado
            emoji = {"Activo": "🟢", "Pasivo": "🟡", "Nuevo": "🔵"}
            clase_css = {"Activo": "resultado-activo", "Pasivo": "resultado-pasivo", "Nuevo": "resultado-nuevo"}
            recomendacion = {
                "Activo": "⭐ Usuario comprometido. Ofrécele recompensas y contenido exclusivo.",
                "Pasivo": "🔔 Usuario en riesgo. Activa campaña de reengagement con notificaciones.",
                "Nuevo": "👋 Usuario explorando. Muéstrale tutoriales y beneficios de la plataforma."
            }

            st.markdown(f"""
            <div class="{clase_css[tipo]}">
                {emoji[tipo]} {tipo.upper()}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Probabilidades
            st.markdown("**📊 Probabilidades:**")
            colores = {"Activo": "#00C9A7", "Pasivo": "#FFB347", "Nuevo": "#4A90D9"}
            for clase, prob in zip(le_target.classes_, probs):
                st.markdown(f"**{clase}**")
                st.progress(float(prob))
                st.markdown(f"_{prob:.1%}_")

            st.markdown("---")
            st.info(recomendacion[tipo])

        else:
            st.markdown("""
            <div style='background:#1A1A2E; border:1px solid #333; border-radius:12px;
                        padding:40px; text-align:center; color:#666;'>
                <h3>👈 Ingresa los datos y presiona<br>Clasificar Usuario</h3>
            </div>
            """, unsafe_allow_html=True)

            # ── PÁGINA 2: DASHBOARD ──
elif pagina == "📊 Dashboard":

    st.markdown("# 📊 Dashboard de Análisis")
    st.markdown("Visualización del dataset y comportamiento del modelo.")
    st.markdown("---")

    import pandas as pd
    df = pd.read_csv('social_media_usage.csv')

    def clasificar_usuario(row):
        if row['Daily_Minutes_Spent'] > 120 and row['Posts_Per_Day'] > 5:
            return 'Activo'
        elif row['Daily_Minutes_Spent'] >= 30 and row['Daily_Minutes_Spent'] <= 120:
            return 'Pasivo'
        else:
            return 'Nuevo'

    df['tipo_usuario'] = df.apply(clasificar_usuario, axis=1)

    col1, col2, col3 = st.columns(3)
    conteos = df['tipo_usuario'].value_counts()

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#00C9A7'>🟢 {conteos.get('Activo', 0)}</h2>
            <p>Usuarios Activos</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#FFB347'>🟡 {conteos.get('Pasivo', 0)}</h2>
            <p>Usuarios Pasivos</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#4A90D9'>🔵 {conteos.get('Nuevo', 0)}</h2>
            <p>Usuarios Nuevos</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("### Distribución de usuarios")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#1A1A2E')
        ax.set_facecolor('#1A1A2E')
        colores = ['#00C9A7', '#FFB347', '#4A90D9']
        barras = ax.bar(conteos.index, conteos.values, color=colores, edgecolor='white', linewidth=0.5)
        ax.set_title('Tipos de Usuario', color='white', fontsize=13)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for b in barras:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
                    str(int(b.get_height())), ha='center', color='white', fontweight='bold')
        st.pyplot(fig)

    with col5:
        st.markdown("### Minutos diarios por tipo")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#1A1A2E')
        ax2.set_facecolor('#1A1A2E')
        for tipo, color in zip(['Activo', 'Pasivo', 'Nuevo'], colores):
            datos_tipo = df[df['tipo_usuario'] == tipo]['Daily_Minutes_Spent']
            ax2.hist(datos_tipo, bins=15, alpha=0.7, color=color, label=tipo, edgecolor='white', linewidth=0.3)
        ax2.set_title('Distribución de Minutos', color='white', fontsize=13)
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('#444')
        ax2.spines['left'].set_color('#444')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(facecolor='#1A1A2E', labelcolor='white')
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### Importancia de variables")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    fig3.patch.set_facecolor('#1A1A2E')
    ax3.set_facecolor('#1A1A2E')
    vars_nombres = ['Daily_Minutes_Spent', 'Likes_Per_Day', 'App_encoded']
    importancias = modelo.feature_importances_
    colores_imp = ['#00C9A7', '#FFB347', '#4A90D9']
    barras3 = ax3.barh(vars_nombres, importancias, color=colores_imp, edgecolor='white', linewidth=0.5)
    ax3.set_title('Variables más importantes', color='white', fontsize=13)
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#444')
    ax3.spines['left'].set_color('#444')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    for b in barras3:
        ax3.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                 f'{b.get_width():.3f}', va='center', color='white', fontweight='bold')
    st.pyplot(fig3)

# ── PÁGINA 3: ACERCA DEL MODELO ──
elif pagina == "ℹ️ Acerca del modelo":

    st.markdown("# ℹ️ Acerca del Modelo")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Detalles técnicos")
        st.markdown("""
        | Parámetro | Valor |
        |-----------|-------|
        | Algoritmo | Random Forest |
        | Árboles | 100 |
        | Profundidad máx. | 10 |
        | Accuracy | 74.5% |
        | Dataset | 1,000 usuarios |
        | Variables | 3 |
        """)

    with col2:
        st.markdown("### 🎯 Categorías del modelo")
        st.markdown("""
        <div style='background:#1A1A2E; padding:16px; border-radius:10px; border:1px solid #333'>
            <p>🟢 <b style='color:#00C9A7'>Activo</b> — Más de 120 min/día con alta interacción</p>
            <p>🟡 <b style='color:#FFB347'>Pasivo</b> — Entre 30 y 120 min/día</p>
            <p>🔵 <b style='color:#4A90D9'>Nuevo</b> — Menos de 30 min/día</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Variables utilizadas")
    st.markdown("""
    - **Daily_Minutes_Spent** — Minutos diarios en la app *(variable más importante: ~65%)*
    - **Likes_Per_Day** — Likes recibidos por día *(importancia secundaria: ~25%)*
    - **App_encoded** — Plataforma social utilizada *(importancia terciaria: ~10%)*
    """)

    st.markdown("---")
    st.info("⚠️ Este modelo fue desarrollado como prueba de concepto aplicable a plataformas de interacción social, usando un dataset público de Kaggle como base de entrenamiento.")