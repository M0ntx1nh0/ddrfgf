# app.py
import streamlit as st
from pathlib import Path
from common.sidebar_branding import render_sidebar_branding
from common.theme import apply_app_theme
from common.image_utils import load_png_with_transparent_white

st.set_page_config(
    page_title="Análisis Wyscout",
    page_icon="📊",
    layout="wide"
)
render_sidebar_branding()
apply_app_theme()

base_dir = Path(__file__).resolve().parent
logo_rfgf = base_dir / "figures" / "images" / "rfgf.png"
logo_curso = base_dir / "figures" / "images" / "Curso_sin.png"

col_logo_1, col_logo_2 = st.columns([1, 1])
with col_logo_1:
    if logo_rfgf.exists():
        st.image(load_png_with_transparent_white(logo_rfgf), width=220)
with col_logo_2:
    if logo_curso.exists():
        st.image(str(logo_curso), width=220)

st.markdown(
    "# Bienvenido a la Aplicación de Scouting\n"
    "### Dirigida a alumnos del II Curso de Dirección Deportiva de la Real Federación Galega de Fútbol"
)

st.markdown(
    """
### 👋 ¿Qué encontrarás en la app?

- 🏆 **Rankings por métricas** para detectar rápido quién destaca en cada aspecto del juego.
- 🔎 **Filtros avanzados** por liga, grupo, posición, minutos, edad y partidos.
- ⭐ **Scoring por perfil** (ofensivo, defensivo y portero/control) con explicación del cálculo.
- 📋 **Tablas de búsqueda por equipo y jugador** con diferencias vs líder y percentiles.
- 🎯 **Scatterplots comparativos** para analizar relaciones entre métricas clave.
- 🕸️ **Radares + swarmplots** para comparar percentiles de jugadores y contexto de distribución.

### ✅ Objetivo
Ayudarte a construir argumentos de scouting sólidos, combinando ranking, contexto y comparación visual.
"""
)
