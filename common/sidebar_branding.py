from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image


@st.cache_data(show_spinner=False)
def _load_sidebar_players() -> pd.DataFrame:
    from common.data_loader import load_players_data

    df = load_players_data().copy()
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def render_sidebar_branding() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    logo_path = base_dir / "figures" / "images" / "MCODE Sport Analytics.png"

    with st.sidebar:
        st.markdown("---")
        st.markdown("🧑‍💻 © App desarrollada por **Ramón Codesido**")
        st.markdown("📧 ramon.codesido@gmail.com")
        if logo_path.exists():
            # Suaviza el fondo blanco del PNG para integrarlo mejor en el sidebar
            with Image.open(logo_path).convert("RGBA") as img:
                data = img.getdata()
                cleaned = []
                for r, g, b, a in data:
                    if r > 245 and g > 245 and b > 245:
                        cleaned.append((r, g, b, 0))
                    else:
                        cleaned.append((r, g, b, a))
                img.putdata(cleaned)
                st.image(img, width=130)

        st.markdown("---")
        st.markdown("### 🔎 Buscador jugador")
        try:
            df_players = _load_sidebar_players()
        except Exception:
            st.caption("No se pudo cargar el buscador de jugadores.")
            return

        if df_players.empty:
            st.caption("Sin datos disponibles.")
            return

        name_col = "full_name" if "full_name" in df_players.columns else "name"
        if name_col not in df_players.columns:
            st.caption("No hay columna de nombre en la data.")
            return

        search_text = st.text_input(
            "Escribe nombre o apellido",
            value="",
            key="sidebar_player_search_text",
            placeholder="Ej: Gabi García",
        ).strip()

        df_search = df_players.copy()
        if search_text:
            df_search = df_search[
                df_search[name_col]
                .astype(str)
                .str.contains(search_text, case=False, na=False)
            ]

        nombres = sorted(df_search[name_col].dropna().astype(str).unique().tolist())
        if not nombres:
            st.caption("No hay coincidencias.")
            return

        jugador_sel = st.selectbox(
            "Selecciona jugador",
            nombres,
            key="sidebar_player_search_select",
        )

        row = df_search[df_search[name_col].astype(str) == jugador_sel].head(1)
        if row.empty:
            return
        player = row.iloc[0]

        image_url = str(player.get("image", "") or "").strip()
        if image_url:
            st.image(image_url, width=110)
        st.markdown(f"**{jugador_sel}**")

        equipo = str(player.get("last_club_name", "-") or "-")
        edad = player.get("age", None)
        edad_txt = "-" if pd.isna(edad) else str(int(edad))
        st.caption(f"Equipo: {equipo}")
        st.caption(f"Edad: {edad_txt}")
