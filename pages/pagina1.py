import streamlit as st
import pandas as pd
from common.data_loader import load_players_data
from common.app_config import is_home_only_mode
from common.sidebar_branding import render_sidebar_branding
from common.theme import apply_app_theme

if is_home_only_mode():
    st.warning("Esta página está deshabilitada en modo alumnos.")
    st.stop()

render_sidebar_branding()
apply_app_theme()

# --- Cargar datos
@st.cache_data
def cargar_datos():
    df = load_players_data()
    df["Group"] = pd.to_numeric(df["Group"], errors="coerce")
    df["minutes_on_field"] = pd.to_numeric(df["minutes_on_field"], errors="coerce").fillna(0)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0)
    return df

df = cargar_datos()
st.title("📋 Informe de Jugador")

# --- Selectores encadenados
categorias = ["Selecciona..."] + sorted(df["League"].dropna().unique().tolist())
categoria = st.selectbox("Selecciona la categoría:", categorias)

grupo = equipo = None

if categoria != "Selecciona...":
    df_filtrado = df[df["League"] == categoria]
    grupos = ["Selecciona..."] + [str(int(g)) for g in sorted(df_filtrado["Group"].dropna().unique())]
    grupo = st.selectbox("Selecciona el grupo:", grupos)

    if grupo != "Selecciona...":
        df_filtrado = df_filtrado[df_filtrado["Group"] == float(grupo)]

    equipos = ["Selecciona..."] + sorted(df_filtrado["last_club_name"].dropna().unique())
    equipo = st.selectbox("Selecciona el equipo:", equipos)
else:
    df_filtrado = df.copy()

# --- Mostrar tabla de jugadores
if equipo and equipo != "Selecciona...":
    jugadores = df_filtrado[df_filtrado["last_club_name"] == equipo].copy()

    orden_posiciones = [
        "Portero", "Lateral", "Defensa central", "Pivote",
        "Mediocampista", "Mediapunta", "Extremo", "Delantero"
    ]
    orden_pos_gen = [
    "Delantero", "Central", "Mediapunta", "Pivote",
    "Lateral", "Extremo", "Portero", "Mediocampista", "Desconocido"
]
    jugadores["position_gen_ESP1"] = pd.Categorical(
        jugadores["position_gen_ESP1"], categories=orden_pos_gen, ordered=True
    )
    jugadores = jugadores.sort_values(by=["position_gen_ESP1", "minutes_on_field"], ascending=[True, False])

    # Selección de columnas y renombrado
    columnas_mostrar = [
        "image", "full_name", "age", "minutes_on_field",
        "primary_position_ESP", "primary_position_percent",
        "secondary_position_ESP", "secondary_position_percent",
        "third_position_ESP", "third_position_percent"
    ]
    jugadores_vista = jugadores[columnas_mostrar].copy()
    jugadores_vista = jugadores_vista.rename(columns={
        "image": "Foto",
        "full_name": "Jugador",
        "age": "Edad",
        "minutes_on_field": "Minutos",
        "primary_position_ESP": "Posición 1",
        "primary_position_percent": "% Pos. 1",
        "secondary_position_ESP": "Posición 2",
        "secondary_position_percent": "% Pos. 2",
        "third_position_ESP": "Posición 3",
        "third_position_percent": "% Pos. 3"
    })

    st.markdown(f"### 👥 Jugadores de **{equipo}**")
    st.dataframe(jugadores_vista.drop(columns=["Foto"]), use_container_width=True)

    # --- Detalles por jugador
    nombres_jugadores = jugadores_vista["Jugador"].tolist()
    jugador_sel = st.selectbox("🔍 Selecciona un jugador para ver detalles:", ["Selecciona..."] + nombres_jugadores)

    if jugador_sel != "Selecciona...":
        jugador_data = jugadores.loc[jugadores["full_name"].str.strip() == jugador_sel.strip()].iloc[0]
        st.markdown(f"## 🧠 Detalles de {jugador_data['full_name']}")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(jugador_data["image"], width=100)
        with col2:
            st.markdown(f"**Edad:** {int(jugador_data['age'])}")
            st.markdown(f"**Minutos jugados:** {int(jugador_data['minutes_on_field'])}")
            st.markdown(f"**Posición principal:** {jugador_data['primary_position_ESP']} ({jugador_data['primary_position_percent']}%)")
            if pd.notna(jugador_data['secondary_position_ESP']):
                st.markdown(f"**Posición secundaria:** {jugador_data['secondary_position_ESP']} ({jugador_data['secondary_position_percent']}%)")
            if pd.notna(jugador_data['third_position_ESP']):
                st.markdown(f"**Tercera posición:** {jugador_data['third_position_ESP']} ({jugador_data['third_position_percent']}%)")
