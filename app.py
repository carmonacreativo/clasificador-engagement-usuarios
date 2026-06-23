import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="Engagement Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_clasificador.pkl')
    le_app = joblib.load('encoder_app.pkl')
    le_target = joblib.load('encoder_target.pkl')
    return modelo, le_app, le_target

modelo, le_app, le_target = cargar_modelo()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #080810 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

[data-testid="stMain"] .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── GLOWS AMBIENTALES ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,201,167,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: glow1 8s ease-in-out infinite alternate;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: -150px; right: -150px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(74,144,217,0.10) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: glow2 10s ease-in-out infinite alternate;
}

@keyframes glow1 {
    0% { transform: translate(0,0) scale(1); opacity: 0.8; }
    100% { transform: translate(80px,60px) scale(1.2); opacity: 1; }
}

@keyframes glow2 {
    0% { transform: translate(0,0) scale(1); opacity: 0.6; }
    100% { transform: translate(-60px,-80px) scale(1.15); opacity: 0.9; }
}

/* ── LAYOUT PRINCIPAL ── */
.dashboard-wrapper {
    display: flex;
    min-height: 100vh;
    position: relative;
    z-index: 1;
}

/* ── SIDEBAR FLOTANTE ── */
.sidebar-float {
    position: fixed;
    left: 16px;
    top: 50%;
    transform: translateY(-50%);
    width: 60px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 16px 10px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    z-index: 100;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.sidebar-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #00C9A7, #4A90D9);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 8px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    letter-spacing: -1px;
}

.nav-icon {
    width: 40px; height: 40px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
    color: rgba(255,255,255,0.4);
    font-size: 18px;
    border: none;
    background: transparent;
}

.nav-icon:hover, .nav-icon.active {
    background: rgba(0,201,167,0.15);
    color: #00C9A7;
    box-shadow: 0 0 20px rgba(0,201,167,0.2);
}

.nav-divider {
    width: 30px; height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 4px 0;
}

/* ── CONTENIDO PRINCIPAL ── */
.main-content {
    margin-left: 90px;
    padding: 32px 32px 32px 0;
    width: 100%;
    min-height: 100vh;
}

/* ── HEADER ── */
.page-header {
    margin-bottom: 28px;
}

.page-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: white;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.page-header p {
    font-size: 14px;
    color: rgba(255,255,255,0.4);
    font-weight: 400;
}

/* ── GLASS CARD ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 24px;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}

/* ── METRIC CARDS ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: -30px; right: -30px;
    width: 80px; height: 80px;
    border-radius: 50%;
    opacity: 0.15;
}

.metric-card.green::after { background: #00C9A7; box-shadow: 0 0 40px #00C9A7; }
.metric-card.yellow::after { background: #FFB347; box-shadow: 0 0 40px #FFB347; }
.metric-card.blue::after { background: #4A90D9; box-shadow: 0 0 40px #4A90D9; }

.metric-label {
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
    color: white;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-sub {
    font-size: 12px;
    color: rgba(255,255,255,0.3);
}

.metric-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

/* ── PREDICTOR GRID ── */
.predictor-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
}

/* ── INPUT SECTION ── */
.input-label {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.input-label::before {
    content: '';
    width: 3px; height: 14px;
    background: #00C9A7;
    border-radius: 2px;
}

/* ── RESULTADO ── */
.resultado-box {
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.resultado-activo {
    background: linear-gradient(135deg, rgba(0,201,167,0.2), rgba(0,201,167,0.05));
    border: 1px solid rgba(0,201,167,0.3);
    box-shadow: 0 0 40px rgba(0,201,167,0.15);
}

.resultado-pasivo {
    background: linear-gradient(135deg, rgba(255,179,71,0.2), rgba(255,179,71,0.05));
    border: 1px solid rgba(255,179,71,0.3);
    box-shadow: 0 0 40px rgba(255,179,71,0.15);
}

.resultado-nuevo {
    background: linear-gradient(135deg, rgba(74,144,217,0.2), rgba(74,144,217,0.05));
    border: 1px solid rgba(74,144,217,0.3);
    box-shadow: 0 0 40px rgba(74,144,217,0.15);
}

.resultado-tipo {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
    opacity: 0.7;
}

.resultado-valor {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 16px;
}

.resultado-espera {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-size: 13px;
    font-weight: 500;
}

/* ── PROB BAR ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.prob-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5);
    width: 50px;
    font-weight: 500;
}

.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}

.prob-pct {
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    width: 38px;
    text-align: right;
    font-weight: 600;
}

/* ── RECOMENDACION ── */
.recomendacion {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    margin-top: 16px;
    line-height: 1.5;
}

/* ── STREAMLIT OVERRIDES ── */
.stSlider > div > div > div { background: rgba(0,201,167,0.3) !important; }
.stSlider > div > div > div > div { background: #00C9A7 !important; }
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: white !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00C9A7, #4A90D9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,201,167,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(0,201,167,0.4) !important;
}

div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stMarkdown p { color: rgba(255,255,255,0.7) !important; }
</style>
""", unsafe_allow_html=True)

# ── ESTADO DE NAVEGACIÓN ──
if 'pagina' not in st.session_state:
    st.session_state.pagina = 'predictor'

# ── SIDEBAR FLOTANTE ──
st.markdown("""
<div class="sidebar-float">
    <div class="sidebar-logo">EC</div>
    <div class="nav-divider"></div>
    <div class="nav-icon active" title="Predictor">⬡</div>
    <div class="nav-icon" title="Dashboard">⬡</div>
    <div class="nav-icon" title="Modelo">⬡</div>
    <div class="nav-divider"></div>
    <div class="nav-icon" title="Info">⬡</div>
</div>
""", unsafe_allow_html=True)

# ── NAVEGACIÓN REAL ──
col_nav1, col_nav2, col_nav3 = st.columns([1,1,1])
with col_nav1:
    if st.button("Predictor", key="nav1"):
        st.session_state.pagina = 'predictor'
with col_nav2:
    if st.button("Dashboard", key="nav2"):
        st.session_state.pagina = 'dashboard'
with col_nav3:
    if st.button("Modelo", key="nav3"):
        st.session_state.pagina = 'modelo'

st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# ════════════════════════════════
# PÁGINA: PREDICTOR
# ════════════════════════════════
if st.session_state.pagina == 'predictor':

    st.markdown("""
    <div class='page-header'>
        <h1>Clasificador de Engagement</h1>
        <p>Ingresa los datos del usuario — el modelo predice su nivel de actividad en tiempo real</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>Datos del usuario</div>", unsafe_allow_html=True)

        minutos = st.slider("Minutos diarios en la app", 5, 500, 120, 5)
        likes = st.slider("Likes recibidos por día", 0, 200, 50, 5)
        app_opciones = list(le_app.classes_)
        app_seleccionada = st.selectbox("Plataforma utilizada", app_opciones)
        predecir = st.button("Clasificar Usuario")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>Resultado</div>", unsafe_allow_html=True)

        if predecir:
            app_encoded = le_app.transform([app_seleccionada])[0]
            datos = pd.DataFrame(
                [[minutos, likes, app_encoded]],
                columns=['Daily_Minutes_Spent', 'Likes_Per_Day', 'App_encoded']
            )
            pred = modelo.predict(datos)[0]
            probs = modelo.predict_proba(datos)[0]
            tipo = le_target.inverse_transform([pred])[0]

            colores_hex = {"Activo": "#00C9A7", "Pasivo": "#FFB347", "Nuevo": "#4A90D9"}
            clase_css = {"Activo": "resultado-activo", "Pasivo": "resultado-pasivo", "Nuevo": "resultado-nuevo"}
            recomendacion = {
                "Activo": "Usuario comprometido — considera recompensas exclusivas y contenido premium para mantener su nivel de actividad.",
                "Pasivo": "Usuario en riesgo de abandono — activa una campaña de reengagement con notificaciones personalizadas.",
                "Nuevo": "Usuario en exploración — muéstrale el valor de la plataforma con tutoriales y beneficios iniciales."
            }

            color = colores_hex[tipo]
            st.markdown(f"""
            <div class='resultado-box {clase_css[tipo]}'>
                <div class='resultado-tipo'>Clasificación del modelo</div>
                <div class='resultado-valor' style='color:{color}'>{tipo}</div>
                <div style='font-size:13px;color:rgba(255,255,255,0.4);margin-bottom:20px'>
                    Accuracy del modelo: 74.5%
                </div>
            """, unsafe_allow_html=True)

            for clase, prob in zip(le_target.classes_, probs):
                c = colores_hex.get(clase, "#888")
                st.markdown(f"""
                <div class='prob-row'>
                    <span class='prob-label'>{clase}</span>
                    <div class='prob-bar-bg'>
                        <div class='prob-bar-fill' style='width:{prob*100:.1f}%;background:{c}'></div>
                    </div>
                    <span class='prob-pct'>{prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='recomendacion'>{recomendacion[tipo]}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class='resultado-espera'>
                Ingresa los parámetros del usuario<br>y presiona Clasificar
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════
# PÁGINA: DASHBOARD
# ════════════════════════════════
elif st.session_state.pagina == 'dashboard':

    df = pd.read_csv('social_media_usage.csv')

    def clasificar(row):
        if row['Daily_Minutes_Spent'] > 120 and row['Posts_Per_Day'] > 5:
            return 'Activo'
        elif row['Daily_Minutes_Spent'] >= 30:
            return 'Pasivo'
        else:
            return 'Nuevo'

    df['tipo_usuario'] = df.apply(clasificar, axis=1)
    conteos = df['tipo_usuario'].value_counts()

    st.markdown("""
    <div class='page-header'>
        <h1>Dashboard de Análisis</h1>
        <p>Visualización del dataset y comportamiento del modelo</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metrics-row'>
        <div class='metric-card green'>
            <div class='metric-label'><span class='metric-dot' style='background:#00C9A7'></span>Activos</div>
            <div class='metric-value'>{conteos.get('Activo',0)}</div>
            <div class='metric-sub'>53.0% del total</div>
        </div>
        <div class='metric-card yellow'>
            <div class='metric-label'><span class='metric-dot' style='background:#FFB347'></span>Pasivos</div>
            <div class='metric-value'>{conteos.get('Pasivo',0)}</div>
            <div class='metric-sub'>26.8% del total</div>
        </div>
        <div class='metric-card blue'>
            <div class='metric-label'><span class='metric-dot' style='background:#4A90D9'></span>Nuevos</div>
            <div class='metric-value'>{conteos.get('Nuevo',0)}</div>
            <div class='metric-sub'>20.2% del total</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2, gap="large")

    def estilo_fig():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.tick_params(colors='rgba(255,255,255,0.4)', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('rgba(255,255,255,0.06)')
        return fig, ax

    with col_g1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>Distribución de usuarios</div>", unsafe_allow_html=True)
        fig, ax = estilo_fig()
        colores = ['#00C9A7', '#FFB347', '#4A90D9']
        barras = ax.bar(conteos.index, conteos.values, color=colores, width=0.5, edgecolor='none')
        for b in barras:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
                    str(int(b.get_height())), ha='center',
                    color='rgba(255,255,255,0.6)', fontsize=10, fontweight='600')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, transparent=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_g2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='input-label'>Importancia de variables</div>", unsafe_allow_html=True)
        fig2, ax2 = estilo_fig()
        vars_n = ['Daily_Minutes', 'Likes_Per_Day', 'App']
        imps = modelo.feature_importances_
        bars2 = ax2.barh(vars_n, imps, color=colores[::-1], height=0.4, edgecolor='none')
        for b in bars2:
            ax2.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                     f'{b.get_width():.3f}', va='center',
                     color='rgba(255,255,255,0.5)', fontsize=10, fontweight='600')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2, transparent=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='input-label'>Distribución de minutos por tipo de usuario</div>", unsafe_allow_html=True)
    fig3, ax3 = estilo_fig()
    fig3.set_size_inches(12, 3)
    for tipo, color in zip(['Activo', 'Pasivo', 'Nuevo'], colores):
        datos_tipo = df[df['tipo_usuario'] == tipo]['Daily_Minutes_Spent']
        ax3.hist(datos_tipo, bins=20, alpha=0.7, color=color, label=tipo, edgecolor='none')
    ax3.legend(facecolor='none', labelcolor='white', framealpha=0, fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    st.pyplot(fig3, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════
# PÁGINA: MODELO
# ════════════════════════════════
elif st.session_state.pagina == 'modelo':

    st.markdown("""
    <div class='page-header'>
        <h1>Acerca del Modelo</h1>
        <p>Detalles técnicos, arquitectura y variables del clasificador</p>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2, gap="large")

    with col_m1:
        st.markdown("""
        <div class='glass-card'>
            <div class='input-label'>Detalles técnicos</div>
            <div style='display:flex;flex-direction:column;gap:12px'>
                <div style='display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                    <span style='color:rgba(255,255,255,0.4);font-size:13px'>Algoritmo</span>
                    <span style='color:white;font-size:13px;font-weight:600'>Random Forest</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                    <span style='color:rgba(255,255,255,0.4);font-size:13px'>Árboles</span>
                    <span style='color:white;font-size:13px;font-weight:600'>100</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                    <span style='color:rgba(255,255,255,0.4);font-size:13px'>Profundidad máx.</span>
                    <span style='color:white;font-size:13px;font-weight:600'>10</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                    <span style='color:rgba(255,255,255,0.4);font-size:13px'>Accuracy</span>
                    <span style='color:#00C9A7;font-size:18px;font-weight:700'>74.5%</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>
                    <span style='color:rgba(255,255,255,0.4);font-size:13px'>Dataset</span>
                    <span style='color:white;font-size:13px;font-weight:600'>1,000 usuarios</span>
                </div>
                <div style='display:flex;justify-content:space-between;padding:10px 0'>
                    <span style='color:rgba(255,255,255,0.4);font-size:13px'>Aprendizaje</span>
                    <span style='color:white;font-size:13px;font-weight:600'>Supervisado</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class='glass-card'>
            <div class='input-label'>Categorías del modelo</div>
            <div style='display:flex;flex-direction:column;gap:12px'>
                <div style='background:rgba(0,201,167,0.08);border:1px solid rgba(0,201,167,0.2);border-radius:12px;padding:16px'>
                    <div style='color:#00C9A7;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px'>Activo</div>
                    <div style='color:rgba(255,255,255,0.5);font-size:13px'>Más de 120 min/día con alta interacción</div>
                </div>
                <div style='background:rgba(255,179,71,0.08);border:1px solid rgba(255,179,71,0.2);border-radius:12px;padding:16px'>
                    <div style='color:#FFB347;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px'>Pasivo</div>
                    <div style='color:rgba(255,255,255,0.5);font-size:13px'>Entre 30 y 120 minutos diarios en la app</div>
                </div>
                <div style='background:rgba(74,144,217,0.08);border:1px solid rgba(74,144,217,0.2);border-radius:12px;padding:16px'>
                    <div style='color:#4A90D9;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px'>Nuevo</div>
                    <div style='color:rgba(255,255,255,0.5);font-size:13px'>Menos de 30 minutos — usuario en exploración</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <div class='input-label'>Variables utilizadas</div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px'>
            <div style='text-align:center;padding:16px;background:rgba(0,201,167,0.06);border-radius:12px;border:1px solid rgba(0,201,167,0.1)'>
                <div style='color:#00C9A7;font-size:28px;font-weight:800;margin-bottom:4px'>65%</div>
                <div style='color:white;font-size:13px;font-weight:600;margin-bottom:4px'>Daily_Minutes_Spent</div>
                <div style='color:rgba(255,255,255,0.3);font-size:11px'>Variable más importante</div>
            </div>
            <div style='text-align:center;padding:16px;background:rgba(255,179,71,0.06);border-radius:12px;border:1px solid rgba(255,179,71,0.1)'>
                <div style='color:#FFB347;font-size:28px;font-weight:800;margin-bottom:4px'>25%</div>
                <div style='color:white;font-size:13px;font-weight:600;margin-bottom:4px'>Likes_Per_Day</div>
                <div style='color:rgba(255,255,255,0.3);font-size:11px'>Variable secundaria</div>
            </div>
            <div style='text-align:center;padding:16px;background:rgba(74,144,217,0.06);border-radius:12px;border:1px solid rgba(74,144,217,0.1)'>
                <div style='color:#4A90D9;font-size:28px;font-weight:800;margin-bottom:4px'>10%</div>
                <div style='color:white;font-size:13px;font-weight:600;margin-bottom:4px'>App_encoded</div>
                <div style='color:rgba(255,255,255,0.3);font-size:11px'>Variable terciaria</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)