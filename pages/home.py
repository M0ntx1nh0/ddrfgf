# pages/home.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import PyPizza
from pathlib import Path
from common.data_loader import load_players_data
from common.sidebar_branding import render_sidebar_branding
from common.theme import apply_app_theme
from common.image_utils import load_png_with_transparent_white

st.set_page_config(layout="wide")
render_sidebar_branding()
apply_app_theme()

@st.cache_data
def load_data():
    df = load_players_data()
    df["Group"] = pd.to_numeric(df["Group"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["total_matches"] = pd.to_numeric(df["total_matches"], errors="coerce")
    return df

df_original = load_data()

base_dir = Path(__file__).resolve().parent.parent
logo_rfgf = base_dir / "figures" / "images" / "rfgf.png"
logo_curso = base_dir / "figures" / "images" / "Curso_sin.png"

col_logo_1, col_logo_2 = st.columns([1, 1])
with col_logo_1:
    if logo_rfgf.exists():
        st.image(load_png_with_transparent_white(logo_rfgf), width=220)
with col_logo_2:
    if logo_curso.exists():
        st.image(str(logo_curso), width=220)

# Métricas base por perfil para scoring/comparativas
metricas_ofensivas = [
    "goals", "head_goals", "non_penalty_goal_avg",
    "xg_shot_avg", "assists_avg", "xg_assist_avg"
]

metricas_defensivas = [
    "duels_won_percent", "interceptions_avg",
    "successful_defensive_actions_avg", "aerial_duels_won",
    "shot_block_avg", "fouls_avg"
]

metricas_porteros = [
    "clean_sheets", "save_percent", "gk_aerial_duels_avg",
    "prevented_goals_avg", "xg_save_avg", "conceded_goals_avg"
]

metricas_control = [
    "passes_avg", "accurate_passes_percent", "progressive_pass_avg",
    "passes_to_final_third_avg", "successful_forward_passes_percent", "successful_long_passes_percent"
]

# ----------------------------
# FUNCIÓN GENERAL DE FILTROS
# ----------------------------
def aplicar_filtros(df, key_prefix="", incluir_filtros_extra=False):
    df_filtrado = df.copy()
    df_league = st.selectbox("Selecciona la liga:", ["Primera RFEF", "Segunda RFEF"], key=f"{key_prefix}_liga")
    df_filtrado = df_filtrado[df_filtrado["League"] == df_league]

    grupos = sorted(df_filtrado["Group"].dropna().unique())
    grupo = st.selectbox("Selecciona el grupo:", ["Total"] + [str(int(g)) for g in grupos], key=f"{key_prefix}_grupo")
    if grupo != "Total":
        df_filtrado = df_filtrado[df_filtrado["Group"] == int(grupo)]

    posiciones = sorted(df_filtrado["position_gen_ESP1"].dropna().unique())
    pos_sel = st.selectbox("Selecciona la posición:", ["Todas las posiciones"] + posiciones, key=f"{key_prefix}_pos")
    if pos_sel != "Todas las posiciones":
        df_filtrado = df_filtrado[df_filtrado["position_gen_ESP1"] == pos_sel]

    max_min = df_filtrado["minutes_on_field"].max()
    df_filtrado["pct_min_jugados"] = (df_filtrado["minutes_on_field"] / max_min * 100).round(1)
    min_pct, max_pct = st.slider("% de minutos jugados", 0, 100, (0, 100), key=f"{key_prefix}_slider")
    df_filtrado = df_filtrado[df_filtrado["pct_min_jugados"].between(min_pct, max_pct)]

    if incluir_filtros_extra and not df_filtrado.empty:
        if "age" in df_filtrado.columns:
            age_series = pd.to_numeric(df_filtrado["age"], errors="coerce").dropna()
            if not age_series.empty:
                age_min_data = int(age_series.min())
                age_max_data = int(age_series.max())
                edad_min, edad_max = st.slider(
                    "Rango de edad",
                    age_min_data,
                    age_max_data,
                    (age_min_data, age_max_data),
                    key=f"{key_prefix}_edad_rango",
                )
                df_filtrado = df_filtrado[df_filtrado["age"].between(edad_min, edad_max, inclusive="both")]

        if "total_matches" in df_filtrado.columns and not df_filtrado.empty:
            match_series = pd.to_numeric(df_filtrado["total_matches"], errors="coerce").dropna()
            if not match_series.empty:
                part_min_data = int(match_series.min())
                part_max_data = int(match_series.max())
                part_min, part_max = st.slider(
                    "Rango de partidos",
                    part_min_data,
                    part_max_data,
                    (part_min_data, part_max_data),
                    key=f"{key_prefix}_partidos_rango",
                )
                df_filtrado = df_filtrado[
                    df_filtrado["total_matches"].between(part_min, part_max, inclusive="both")
                ]

    return df_filtrado


def calcular_scoring_perfiles(df_input: pd.DataFrame) -> pd.DataFrame:
    df_norm = df_input.copy()
    metricas_all = metricas_ofensivas + metricas_defensivas + metricas_porteros + metricas_control

    for col in metricas_all:
        if col not in df_norm.columns:
            continue
        serie = pd.to_numeric(df_norm[col], errors="coerce")
        min_val = serie.min()
        max_val = serie.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            df_norm[col + "_norm"] = 0.0
        elif col == "conceded_goals_avg":
            df_norm[col + "_norm"] = 1 - (serie - min_val) / (max_val - min_val)
        else:
            df_norm[col + "_norm"] = (serie - min_val) / (max_val - min_val)

    df_norm["scoring_ofensivo"] = df_norm[
        [c + "_norm" for c in metricas_ofensivas if c + "_norm" in df_norm.columns]
    ].mean(axis=1)
    df_norm["scoring_defensivo"] = df_norm[
        [c + "_norm" for c in metricas_defensivas if c + "_norm" in df_norm.columns]
    ].mean(axis=1)
    df_norm["scoring_portero"] = df_norm[
        [c + "_norm" for c in metricas_porteros if c + "_norm" in df_norm.columns]
    ].mean(axis=1)
    df_norm["scoring_control"] = df_norm[
        [c + "_norm" for c in metricas_control if c + "_norm" in df_norm.columns]
    ].mean(axis=1)
    df_norm["scoring_portero_control"] = np.where(
        df_norm["position_gen_ESP1"].astype(str) == "Portero",
        df_norm["scoring_portero"],
        df_norm["scoring_control"],
    )
    df_norm["scoring_total"] = df_norm[
        ["scoring_ofensivo", "scoring_defensivo", "scoring_portero_control"]
    ].mean(axis=1)
    return df_norm


def render_scoring_lookup(df_base: pd.DataFrame, key_prefix: str, section_title: str):
    st.markdown(section_title)
    with st.expander("ℹ️ ¿Cómo leer esta tabla de Scoring?"):
        st.markdown(
            """
- `Score Ofensivo`, `Score Defensivo` y `Score Portero/Control`: rendimiento relativo por perfil.
- `Score Total`: promedio de perfiles (Ofensivo + Defensivo + Portero/Control).
- `Dif ... vs 1º`: diferencia frente al líder de ese ranking en la muestra filtrada.
- `Percentil en su posición`: situación del jugador respecto a jugadores de su misma posición (0-100).

**¿Qué significa `Score Portero/Control` y por qué se llama así?**
- Para jugadores con posición `Portero`, esta columna usa métricas específicas de portero (`Score Portero`).
- Para el resto de posiciones, usa métricas de construcción y gestión del juego (`Score Control`).
- Se muestra en una sola columna para mantener una estructura común de tabla y un `Score Total` comparable entre perfiles.

✅ Verde = mejor situación relativa; 🔴 rojo = más lejos del líder o percentil bajo.
"""
        )
    if df_base.empty:
        st.info("No hay datos en la muestra actual para construir la comparativa.")
        return

    df_scores = calcular_scoring_perfiles(df_base)
    df_scores["percentil_posicion"] = (
        df_scores.groupby("position_gen_ESP1")["scoring_total"].rank(pct=True) * 100
    )

    # Referencias contra el líder de cada ranking en la muestra filtrada
    ref_of = pd.to_numeric(df_scores["scoring_ofensivo"], errors="coerce").max()
    ref_df = pd.to_numeric(df_scores["scoring_defensivo"], errors="coerce").max()
    ref_po = pd.to_numeric(
        df_scores.loc[df_scores["position_gen_ESP1"] == "Portero", "scoring_portero"],
        errors="coerce",
    ).max()
    ref_ct = pd.to_numeric(
        df_scores.loc[df_scores["position_gen_ESP1"] != "Portero", "scoring_control"],
        errors="coerce",
    ).max()
    ref_tt = pd.to_numeric(df_scores["scoring_total"], errors="coerce").max()

    equipos = sorted(df_scores["last_club_name"].dropna().unique().tolist())
    equipo_sel = st.selectbox("Equipo", ["Seleccionar..."] + equipos, key=f"{key_prefix}_equipo")

    if equipo_sel != "Seleccionar...":
        df_equipo = df_scores[df_scores["last_club_name"] == equipo_sel].copy()
        jugadores_equipo = sorted(df_equipo["name"].dropna().unique().tolist())
        top5_equipo = (
            df_equipo.sort_values("scoring_total", ascending=False)["name"]
            .dropna()
            .head(5)
            .tolist()
        )
        jugadores_sel = st.multiselect(
            "Jugadores del equipo",
            jugadores_equipo,
            default=top5_equipo if top5_equipo else jugadores_equipo[: min(5, len(jugadores_equipo))],
            key=f"{key_prefix}_jugadores",
        )

        if not jugadores_sel:
            st.info("Selecciona al menos un jugador para mostrar la tabla comparativa.")
        else:
            df_comp = df_equipo[df_equipo["name"].isin(jugadores_sel)].copy()
            df_comp = df_comp.sort_values("scoring_total", ascending=False).reset_index(drop=True)
            df_comp["dif_scoring_ofensivo_vs_1o"] = df_comp["scoring_ofensivo"] - ref_of
            df_comp["dif_scoring_defensivo_vs_1o"] = df_comp["scoring_defensivo"] - ref_df
            df_comp["dif_scoring_portero_control_vs_1o"] = np.where(
                df_comp["position_gen_ESP1"] == "Portero",
                df_comp["scoring_portero"] - ref_po,
                df_comp["scoring_control"] - ref_ct,
            )
            df_comp["dif_scoring_total_vs_1o"] = df_comp["scoring_total"] - ref_tt

            tabla = df_comp[
                [
                    "name",
                    "last_club_name",
                    "age",
                    "position_gen_ESP1",
                    "scoring_ofensivo",
                    "dif_scoring_ofensivo_vs_1o",
                    "scoring_defensivo",
                    "dif_scoring_defensivo_vs_1o",
                    "scoring_portero_control",
                    "dif_scoring_portero_control_vs_1o",
                    "scoring_total",
                    "dif_scoring_total_vs_1o",
                    "percentil_posicion",
                ]
            ].copy()

            tabla = tabla.rename(
                columns={
                    "name": "Jugador",
                    "last_club_name": "Equipo",
                    "age": "Edad",
                    "position_gen_ESP1": "Posición",
                    "scoring_ofensivo": "Score Ofensivo",
                    "dif_scoring_ofensivo_vs_1o": "Dif Ofensivo vs 1º",
                    "scoring_defensivo": "Score Defensivo",
                    "dif_scoring_defensivo_vs_1o": "Dif Defensivo vs 1º",
                    "scoring_portero_control": "Score Portero/Control",
                    "dif_scoring_portero_control_vs_1o": "Dif Portero/Control vs 1º",
                    "scoring_total": "Score Total",
                    "dif_scoring_total_vs_1o": "Dif Total vs 1º",
                    "percentil_posicion": "Percentil en su posición",
                }
            )

            cols_num = [
                "Score Ofensivo", "Dif Ofensivo vs 1º",
                "Score Defensivo", "Dif Defensivo vs 1º",
                "Score Portero/Control", "Dif Portero/Control vs 1º",
                "Score Total", "Dif Total vs 1º",
                "Percentil en su posición",
            ]
            for c in cols_num:
                tabla[c] = pd.to_numeric(tabla[c], errors="coerce").round(2)

            cols_dif = [
                "Dif Ofensivo vs 1º",
                "Dif Defensivo vs 1º",
                "Dif Portero/Control vs 1º",
                "Dif Total vs 1º",
            ]
            tabla_styled = (
                tabla.style
                .format(
                    {
                        "Score Ofensivo": "{:.2f}",
                        "Score Defensivo": "{:.2f}",
                        "Score Portero/Control": "{:.2f}",
                        "Score Total": "{:.2f}",
                        "Dif Ofensivo vs 1º": "{:+.2f}",
                        "Dif Defensivo vs 1º": "{:+.2f}",
                        "Dif Portero/Control vs 1º": "{:+.2f}",
                        "Dif Total vs 1º": "{:+.2f}",
                        "Percentil en su posición": "{:.1f}",
                    }
                )
                .background_gradient(subset=["Percentil en su posición"], cmap="RdYlGn")
                .background_gradient(subset=cols_dif, cmap="RdYlGn")
            )

            st.dataframe(tabla_styled, use_container_width=True)
            st.caption("Las diferencias vs 1º se calculan contra el líder de cada ranking (Ofensivo, Defensivo, Portero y Total) dentro de la muestra filtrada.")



def render_rank_lookup(df_base: pd.DataFrame, key_prefix: str, section_title: str):
    st.markdown(section_title)
    with st.expander("ℹ️ ¿Cómo leer esta tabla de Rankings?"):
        st.markdown(
            """
- `Valor (métrica)`: dato bruto del jugador en la métrica elegida.
- `Posición en ranking`: puesto del jugador en esa métrica (1 = mejor).
- `Dif valor vs 1º`: distancia respecto al valor del líder del ranking.
- `Percentil métrica`: posición relativa del jugador en esa métrica dentro de la muestra (0-100).

📌 La referencia siempre es la muestra filtrada actual (liga, grupo, posición, minutos, edad y partidos).
"""
        )
    if df_base.empty:
        st.info("No hay datos en la muestra actual para construir la comparativa de ranking.")
        return

    metricas_rank = [
        "goals", "head_goals", "non_penalty_goal_avg", "xg_shot_avg", "assists_avg", "xg_assist_avg",
        "defensive_duels_won", "interceptions_avg", "successful_defensive_actions_avg", "aerial_duels_won",
        "shot_block_avg", "fouls_avg",
        "clean_sheets", "save_percent", "gk_aerial_duels_avg", "prevented_goals_avg", "xg_save_avg", "conceded_goals_avg",
    ]
    metricas_rank = [m for m in metricas_rank if m in df_base.columns]
    if not metricas_rank:
        st.info("No hay métricas de ranking disponibles.")
        return

    metrica_sel = st.selectbox("Métrica de ranking", metricas_rank, key=f"{key_prefix}_metrica")
    asc = metrica_sel == "conceded_goals_avg"

    df_rank_m = df_base.copy()
    df_rank_m[metrica_sel] = pd.to_numeric(df_rank_m[metrica_sel], errors="coerce")
    df_rank_m = df_rank_m.dropna(subset=[metrica_sel]).copy()
    if df_rank_m.empty:
        st.info("No hay datos válidos para la métrica seleccionada.")
        return

    df_rank_m["posicion_ranking"] = df_rank_m[metrica_sel].rank(method="min", ascending=asc)
    if asc:
        df_rank_m["percentil_metrica"] = (1 - df_rank_m[metrica_sel].rank(pct=True, ascending=True)) * 100
    else:
        df_rank_m["percentil_metrica"] = df_rank_m[metrica_sel].rank(pct=True, ascending=True) * 100

    equipo_sel = st.selectbox(
        "Equipo",
        ["Seleccionar..."] + sorted(df_rank_m["last_club_name"].dropna().unique().tolist()),
        key=f"{key_prefix}_equipo",
    )
    top_val = df_rank_m[metrica_sel].min() if asc else df_rank_m[metrica_sel].max()

    if equipo_sel != "Seleccionar...":
        df_eq = df_rank_m[df_rank_m["last_club_name"] == equipo_sel].copy()
        if df_eq.empty:
            st.info("No hay jugadores de ese equipo para esta métrica.")
        else:
            top5_eq = df_eq.sort_values("posicion_ranking", ascending=True)["name"].head(5).tolist()
            jugadores_sel = st.multiselect(
                "Jugadores del equipo",
                sorted(df_eq["name"].dropna().unique().tolist()),
                default=top5_eq,
                key=f"{key_prefix}_jugadores",
            )
            if not jugadores_sel:
                st.info("Selecciona al menos un jugador.")
            else:
                df_comp = df_eq[df_eq["name"].isin(jugadores_sel)].copy()
                df_comp = df_comp.sort_values("posicion_ranking", ascending=True).reset_index(drop=True)
                df_comp["dif_valor_vs_1o"] = (top_val - df_comp[metrica_sel]) if asc else (df_comp[metrica_sel] - top_val)

                tabla = df_comp[
                    [
                        "name", "last_club_name", "age", "position_gen_ESP1",
                        metrica_sel, "posicion_ranking", "dif_valor_vs_1o", "percentil_metrica"
                    ]
                ].copy().rename(
                    columns={
                        "name": "Jugador",
                        "last_club_name": "Equipo",
                        "age": "Edad",
                        "position_gen_ESP1": "Posición",
                        metrica_sel: f"Valor ({metrica_sel})",
                        "posicion_ranking": "Posición en ranking",
                        "dif_valor_vs_1o": "Dif valor vs 1º",
                        "percentil_metrica": "Percentil métrica",
                    }
                )

                tabla["Posición en ranking"] = pd.to_numeric(tabla["Posición en ranking"], errors="coerce").astype("Int64")
                for c in [f"Valor ({metrica_sel})", "Dif valor vs 1º", "Percentil métrica"]:
                    tabla[c] = pd.to_numeric(tabla[c], errors="coerce").round(2)

                tabla_styled = (
                    tabla.style
                    .format(
                        {
                            f"Valor ({metrica_sel})": "{:.2f}",
                            "Dif valor vs 1º": "{:+.2f}",
                            "Percentil métrica": "{:.1f}",
                        }
                    )
                    .background_gradient(subset=["Percentil métrica"], cmap="RdYlGn")
                    .background_gradient(subset=["Dif valor vs 1º"], cmap="RdYlGn")
                )
                st.dataframe(tabla_styled, use_container_width=True)
                st.caption("En Ranking, la referencia es el 1º de la métrica seleccionada en la muestra filtrada.")


# =============================
# TABS HOME
# =============================
tab_rankings_home, tab_scoring_home, tab_graficos_home = st.tabs(["🏆 Rankings", "⭐ Scoring", "📊 Gráficos"])

with tab_rankings_home:
    # =============================
    # RANKINGS
    # =============================
    st.markdown("## 🏆 Rankings de estadísticas")
    with st.expander("ℹ️ ¿Qué son los Rankings y cómo leerlos?"):
        st.markdown(
            """
Los rankings ordenan jugadores por métrica dentro de los filtros activos (liga, grupo, posición, minutos, edad y partidos).  

- 🟢 **Top ranking**: jugadores con mejor rendimiento relativo en cada métrica.
- 🎯 **Uso recomendado**: detectar perfiles destacados rápidamente antes de pasar a comparativas más profundas (scoring, scatter y radar).
- ⚠️ **Importante**: el puesto en ranking depende del grupo filtrado, no es una nota absoluta.
"""
        )
    df_rank = aplicar_filtros(df_original, key_prefix="rank", incluir_filtros_extra=True)

    def mostrar_rankings(metricas):
        row_size = 6
        for r in range(0, len(metricas), row_size):
            cols = st.columns(row_size)
            for i, col_name in enumerate(metricas[r:r + row_size]):
                if col_name not in df_rank.columns:
                    continue
                subdf = df_rank[["name", "last_club_name", "age", "image", "primary_position_ESP", col_name]].dropna(subset=[col_name])
                subdf = subdf.sort_values(col_name, ascending=(col_name == "conceded_goals_avg")).head(10)
                with cols[i]:
                    st.markdown(f"#### {col_name.replace('_avg','').replace('_percent',' %').replace('_', ' ').capitalize()}")
                    for _, row in subdf.iterrows():
                        st.image(row["image"], width=50)
                        st.markdown(f"**{row['name']}** ({row['last_club_name']})")
                        st.markdown(f"🧩 Posición: *{row['primary_position_ESP']}*")
                        st.markdown(f"<span style='background-color:#D4EDDA;padding:2px 6px;border-radius:4px;'>Edad: {int(row['age'])}</span>", unsafe_allow_html=True)
                        st.write(f"{row[col_name]:.2f}")
            st.divider()

    tab1, tab2, tab3 = st.tabs(["⚽ Ofensivas", "🛡️ Defensivas", "🧤 Porteros"])
    with tab1:
        st.markdown("<h2 style='text-align:center;'>⚽ Ofensivas</h2>", unsafe_allow_html=True)
        mostrar_rankings(["goals", "head_goals", "non_penalty_goal_avg", "xg_shot_avg", "assists_avg", "xg_assist_avg"])
    with tab2:
        st.markdown("<h3 style='text-align:center;'>🛡️ Defensivas</h3>", unsafe_allow_html=True)
        mostrar_rankings(["defensive_duels_won", "interceptions_avg", "successful_defensive_actions_avg", "aerial_duels_won", "shot_block_avg", "fouls_avg"])
    with tab3:
        st.markdown("<h3 style='text-align:center;'>🧤 Porteros</h3>", unsafe_allow_html=True)
        mostrar_rankings(["clean_sheets", "save_percent", "gk_aerial_duels_avg", "prevented_goals_avg", "xg_save_avg", "conceded_goals_avg"])

    render_rank_lookup(df_base=df_rank, key_prefix="rank_lookup", section_title="### 🔎 Búsqueda de jugador (contexto Rankings)")

with tab_scoring_home:
    # =============================
    # SCORING POR PERFIL
    # =============================
    st.markdown("## ⭐ Scoring por perfil de jugador")
    with st.expander("¿Cómo se calcula el scoring?"):
        st.markdown(
            """
    El scoring es una **nota comparativa** para ordenar jugadores dentro de los filtros que elijas (liga, grupo, posición y minutos).

    **Pasos del cálculo:**
    1. Se seleccionan métricas por perfil (ofensivo, defensivo o portero).
    2. Cada métrica se **normaliza** entre 0 y 1 (min-max) dentro del grupo filtrado.
    3. En métricas donde **menos es mejor** (por ejemplo, `conceded_goals_avg`), la escala se invierte.
    4. Se calcula la **media** de esas métricas normalizadas para obtener el score final.

    **Cómo interpretarlo:**
    - Cuanto más alto el score, mejor rendimiento relativo en ese perfil.
    - No es una nota absoluta universal: depende del grupo que estés comparando.
    """
        )
    df_scoring = aplicar_filtros(df_original, key_prefix="scoring")

    df_norm = calcular_scoring_perfiles(df_scoring)

    # Scoring ofensivo
    ofensivos = df_norm[df_norm["position_gen_ESP1"] != "Portero"].copy()
    ofensivos["scoring_ofensivo"] = ofensivos[
        [c + "_norm" for c in metricas_ofensivas if c + "_norm" in ofensivos]
    ].mean(axis=1)
    top_ofensivos = ofensivos.sort_values("scoring_ofensivo", ascending=False).head(10)

    # Scoring defensivo
    defensivos = df_norm[df_norm["position_gen_ESP1"] != "Portero"].copy()
    defensivos["scoring_defensivo"] = defensivos[
        [c + "_norm" for c in metricas_defensivas if c + "_norm" in defensivos]
    ].mean(axis=1)
    top_defensivos = defensivos.sort_values("scoring_defensivo", ascending=False).head(10)

    # Scoring porteros
    porteros = df_norm[df_norm["position_gen_ESP1"] == "Portero"].copy()
    porteros["scoring_portero"] = porteros[
        [c + "_norm" for c in metricas_porteros if c + "_norm" in porteros]
    ].mean(axis=1)
    top_porteros = porteros.sort_values("scoring_portero", ascending=False).head(10)

    # Mostrar rankings
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚽ Top 10 Ofensivos")
        for _, row in top_ofensivos.iterrows():
            st.image(row["image"], width=50)
            st.markdown(f"**{row['name']}** ({row['last_club_name']})")
            st.markdown(f"🧩 Posición: *{row['primary_position_ESP']}*")
            st.write(f"Edad: {int(row['age'])} | Score: {row['scoring_ofensivo']:.2f}")

    with col2:
        st.subheader("🛡️ Top 10 Defensivos")
        for _, row in top_defensivos.iterrows():
            st.image(row["image"], width=50)
            st.markdown(f"**{row['name']}** ({row['last_club_name']})")
            st.markdown(f"🧩 Posición: *{row['primary_position_ESP']}*")
            st.write(f"Edad: {int(row['age'])} | Score: {row['scoring_defensivo']:.2f}")

    with col3:
        st.subheader("🧤 Top 10 Porteros")
        for _, row in top_porteros.iterrows():
            st.image(row["image"], width=50)
            st.markdown(f"**{row['name']}** ({row['last_club_name']})")
            st.markdown(f"🧩 Posición: *{row['primary_position_ESP']}*")
            st.write(f"Edad: {int(row['age'])} | Score: {row['scoring_portero']:.2f}")

    render_scoring_lookup(df_base=df_scoring, key_prefix="scoring_lookup", section_title="### 🔎 Búsqueda de jugador (contexto Scoring)")


with tab_graficos_home:
    # =============================
    # SCATTER PLOTS
    # =============================
    st.markdown("## 🎯 Gráficos de dispersión con destacados")
    df_scatter = aplicar_filtros(df_original, key_prefix="scatter", incluir_filtros_extra=True)

    # Checkbox para excluir valores cero
    excluir_ceros = st.checkbox("❌ Excluir jugadores con valor 0 en alguna métrica seleccionada", value=True)

    def scatter_plot(df, x, y, title):
        df_plot = df.copy()

        # Excluir ceros si el checkbox está activo
        if excluir_ceros:
            df_plot = df_plot[(df_plot[x] != 0) & (df_plot[y] != 0)]

        fig = px.scatter(
            df_plot, x=x, y=y,
            hover_name="name",
            hover_data={"last_club_name": True},
            opacity=0.6,
            color_discrete_sequence=["#1f77b4"]
        )
        fig.add_vline(x=df_plot[x].mean(), line_dash="dash", line_color="gray")
        fig.add_hline(y=df_plot[y].mean(), line_dash="dash", line_color="gray")

        top_x = df_plot.sort_values(x, ascending=False).head(3)
        top_y = df_plot.sort_values(y, ascending=False).head(3)
        top_xy = df_plot.sort_values([x, y], ascending=[False, False]).head(3)
        destacados = pd.concat([top_x, top_y, top_xy]).drop_duplicates()

        for _, row in destacados.iterrows():
            fig.add_annotation(
                x=row[x], y=row[y],
                text=f"{row['name']}<br>({row['last_club_name']})",
                showarrow=True, arrowhead=1,
                font=dict(size=10, color="black")
            )

        jugadores = df_plot.sort_values([x, y], ascending=False)["name"].unique().tolist()
        jugadores_sel = st.multiselect(f"Selecciona jugadores a destacar en '{title}':", jugadores)

        for nombre in jugadores_sel:
            jugador_row = df_plot[df_plot["name"] == nombre].iloc[0]
            fig.add_trace(go.Scatter(
                x=[jugador_row[x]], y=[jugador_row[y]],
                mode="markers+text", name=nombre,
                marker=dict(size=12, color="red"),
                text=[f"{nombre} ({jugador_row['last_club_name']})"],
                textposition="top center"
            ))

        fig.update_layout(height=500, margin=dict(t=40, b=20, l=0, r=0), title=title)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ ¿Qué es un scatterplot y qué métricas puedo combinar?"):
        st.markdown(
            """
Un **scatterplot** (gráfico de dispersión) muestra la relación entre dos métricas:
- Eje X: métrica 1
- Eje Y: métrica 2

Cada punto es un jugador. Te permite detectar perfiles, outliers y comparaciones rápidas.

**Ejemplos interesantes por posición:**
- 🥅 **Portero**:
  1. `save_percent` vs `xg_save_avg`
  2. `prevented_goals_avg` vs `conceded_goals_avg`
  3. `gk_aerial_duels_avg` vs `shots_against_avg`
- 🛡️ **Defensa**:
  1. `defensive_duels_won` vs `interceptions_avg`
  2. `aerial_duels_won` vs `aerial_duels_avg`
  3. `successful_defensive_actions_avg` vs `tackle_avg`
- 🎛️ **Centrocampista**:
  1. `passes_avg` vs `accurate_passes_percent`
  2. `progressive_pass_avg` vs `passes_to_final_third_avg`
  3. `successful_forward_passes_percent` vs `xg_assist_avg`
- ⚽ **Delantero**:
  1. `non_penalty_goal_avg` vs `xg_shot_avg`
  2. `assists_avg` vs `xg_assist_avg`
  3. `shots_on_target_percent` vs `offensive_duels_won`
"""
        )

    st.markdown("## 📊 Scatterplots editables")
    metricas_numericas = df_scatter.select_dtypes(include="number").columns.tolist()
    excluir = ["id", "minutes_on_field"]
    metricas_numericas = sorted([m for m in metricas_numericas if m not in excluir])

    opciones = ["Seleccionar..."] + metricas_numericas

    s1_col1, s1_col2 = st.columns(2)
    with s1_col1:
        x1 = st.selectbox("Scatterplot 1 - Eje X", opciones, index=opciones.index("non_penalty_goal_avg") if "non_penalty_goal_avg" in opciones else 0, key="scatter_1_x")
    with s1_col2:
        y1 = st.selectbox("Scatterplot 1 - Eje Y", opciones, index=opciones.index("xg_shot_avg") if "xg_shot_avg" in opciones else 0, key="scatter_1_y")
    if x1 != "Seleccionar..." and y1 != "Seleccionar...":
        scatter_plot(df_scatter, x1, y1, "Scatterplot 1")

    s2_col1, s2_col2 = st.columns(2)
    with s2_col1:
        x2 = st.selectbox("Scatterplot 2 - Eje X", opciones, index=opciones.index("assists_avg") if "assists_avg" in opciones else 0, key="scatter_2_x")
    with s2_col2:
        y2 = st.selectbox("Scatterplot 2 - Eje Y", opciones, index=opciones.index("xg_assist_avg") if "xg_assist_avg" in opciones else 0, key="scatter_2_y")
    if x2 != "Seleccionar..." and y2 != "Seleccionar...":
        scatter_plot(df_scatter, x2, y2, "Scatterplot 2")

    s3_col1, s3_col2 = st.columns(2)
    with s3_col1:
        x3 = st.selectbox("Scatterplot 3 - Eje X", opciones, index=opciones.index("shots_on_target_percent") if "shots_on_target_percent" in opciones else 0, key="scatter_3_x")
    with s3_col2:
        y3 = st.selectbox("Scatterplot 3 - Eje Y", opciones, index=opciones.index("offensive_duels_won") if "offensive_duels_won" in opciones else 0, key="scatter_3_y")
    if x3 != "Seleccionar..." and y3 != "Seleccionar...":
        scatter_plot(df_scatter, x3, y3, "Scatterplot 3")

    # =============================
    # RADAR CHART (mplsoccer)
    # =============================
    st.markdown("## 🕸️ Radar chart (mplsoccer) con percentiles")
    df_radar = aplicar_filtros(df_original, key_prefix="radar")


    def posicion_general_4(pos_gen):
        pos = str(pos_gen).strip().lower()
        if "portero" in pos:
            return "Portero"
        if any(k in pos for k in ["defensa", "central", "lateral"]):
            return "Defensa"
        if any(k in pos for k in ["delantero", "extremo"]):
            return "Delantero"
        if any(k in pos for k in ["medio", "pivote", "interior"]):
            return "Centrocampista"
        return "Centrocampista"


    def percentil_valor(serie, valor):
        s = pd.to_numeric(serie, errors="coerce").dropna()
        if s.empty or pd.isna(valor):
            return 0.0
        return float((s <= float(valor)).mean() * 100.0)


    def percentiles_jugador(df_ref, jugador_row, metricas):
        vals = []
        for m in metricas:
            jugador_val = pd.to_numeric(pd.Series([jugador_row[m]]), errors="coerce").iloc[0]
            vals.append(percentil_valor(df_ref[m], jugador_val))
        return vals


    def percentiles_promedio(df_ref, metricas):
        vals = []
        for m in metricas:
            serie = pd.to_numeric(df_ref[m], errors="coerce")
            media = serie.mean(skipna=True)
            vals.append(percentil_valor(serie, media))
        return vals


    def nombre_metrica_legible(m):
        return (
            m.replace("_avg", "")
            .replace("_percent", " %")
            .replace("_", " ")
            .title()
        )


    def crear_radar_pizza(metricas, valores_a, valores_b, nombre_a, nombre_b, titulo):
        params = [nombre_metrica_legible(m) for m in metricas]
        baker = PyPizza(
            params=params,
            background_color="#ffffff",
            straight_line_color="#222222",
            straight_line_lw=1,
            last_circle_color="#222222",
            last_circle_lw=1,
            other_circle_ls="-.",
            other_circle_lw=1,
        )

        fig, _ = baker.make_pizza(
            [round(v, 1) for v in valores_a],
            compare_values=[round(v, 1) for v in valores_b],
            figsize=(5.8, 5.8),
            kwargs_slices=dict(facecolor="#1f77b4", edgecolor="#222222", linewidth=1.5, alpha=0.65),
            kwargs_compare=dict(facecolor="#ff7f0e", edgecolor="#222222", linewidth=1.5, alpha=0.60),
            kwargs_params=dict(color="#111111", fontsize=8, va="center"),
            kwargs_values=dict(
                color="#111111",
                fontsize=7,
                bbox=dict(edgecolor="#1f77b4", facecolor="#dceeff", boxstyle="round,pad=0.2", lw=1),
            ),
            kwargs_compare_values=dict(
                color="#111111",
                fontsize=7,
                bbox=dict(edgecolor="#ff7f0e", facecolor="#ffe6cc", boxstyle="round,pad=0.2", lw=1),
            ),
        )

        fig.text(0.5, 0.95, titulo, ha="center", fontsize=10, fontweight="bold")
        fig.text(0.08, 0.03, f"Azul: {nombre_a}", fontsize=7, color="#1f77b4")
        fig.text(0.08, 0.00, f"Naranja: {nombre_b}", fontsize=7, color="#ff7f0e")
        return fig


    def format_market_value(value):
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(v):
            return "N/D"
        if v >= 1_000_000:
            return f"EUR {v / 1_000_000:.1f}M"
        return f"EUR {int(v):,}".replace(",", ".")


    def player_info_label(row):
        return (
            f"{row['name']} | {row['last_club_name']} | "
            f"Edad {int(pd.to_numeric(pd.Series([row['age']]), errors='coerce').fillna(0).iloc[0])} | "
            f"Valor {format_market_value(row.get('market_value'))}"
        )


    def render_three_radars(df_all, df_ref, row_base, row_cmp, cmp_label, referencia_txt, categorias):
        info_base = player_info_label(row_base)
        col1, col2, col3 = st.columns(3)
        col_map = {"Ofensivo": col1, "Defensivo": col2, "Control": col3}

        q_colors = {
            "Q1": "#d73027",
            "Q2": "#fc8d59",
            "Q3": "#fee08b",
            "Q4": "#1a9850",
        }

        def quartile_label(series, value):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty or pd.isna(value):
                return "Q1"
            q1, q2, q3 = s.quantile([0.25, 0.50, 0.75]).values
            if value <= q1:
                return "Q1"
            if value <= q2:
                return "Q2"
            if value <= q3:
                return "Q3"
            return "Q4"

        def crear_swarmplots_categoria(metricas, categoria, jugador_a, jugador_b=None):
            fig, axes = plt.subplots(6, 1, figsize=(6.2, 12.8))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            rng = np.random.default_rng(42)

            for i, metrica in enumerate(metricas[:6]):
                ax = axes[i]
                serie = pd.to_numeric(df_ref[metrica], errors="coerce")
                data_plot = df_ref.loc[serie.notna(), ["name", "last_club_name", "age", metrica]].copy()

                if data_plot.empty:
                    ax.set_title(nombre_metrica_legible(metrica), fontsize=8)
                    ax.axis("off")
                    continue

                data_plot["quartil"] = data_plot[metrica].apply(lambda v: quartile_label(data_plot[metrica], v))
                data_plot["color"] = data_plot["quartil"].map(q_colors)
                data_plot = data_plot.reset_index(drop=True)

                jitter = rng.uniform(-0.12, 0.12, len(data_plot))
                data_plot["y_jitter"] = jitter
                ax.scatter(
                    data_plot[metrica].values,
                    jitter,
                    c=data_plot["color"].values,
                    s=24,
                    alpha=0.85,
                    edgecolors="none",
                )

                val_a = pd.to_numeric(pd.Series([jugador_a.get(metrica)]), errors="coerce").iloc[0]
                if not pd.isna(val_a):
                    ax.scatter([val_a], [0.00], c="#1f77b4", s=95, edgecolors="black", linewidths=0.8, zorder=6)
                    ax.annotate(
                        str(jugador_a.get("name", "Jugador base")),
                        xy=(val_a, 0.00),
                        xytext=(val_a, 0.26),
                        textcoords="data",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="#1f77b4",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#e7f1ff", ec="#1f77b4", lw=1),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="#1f77b4",
                            lw=0.9,
                            connectionstyle="arc3,rad=-0.25",
                        ),
                    )

                if jugador_b is not None:
                    val_b = pd.to_numeric(pd.Series([jugador_b.get(metrica)]), errors="coerce").iloc[0]
                    if not pd.isna(val_b):
                        ax.scatter([val_b], [0.06], c="#ff7f0e", s=95, edgecolors="black", linewidths=0.8, zorder=6)
                        ax.annotate(
                            str(jugador_b.get("name", "Comparador")),
                            xy=(val_b, 0.06),
                            xytext=(val_b, -0.24),
                            textcoords="data",
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color="#ff7f0e",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#fff1e6", ec="#ff7f0e", lw=1),
                            arrowprops=dict(
                                arrowstyle="->",
                                color="#ff7f0e",
                                lw=0.9,
                                connectionstyle="arc3,rad=0.25",
                            ),
                        )

                idx_max = data_plot[metrica].astype(float).idxmax()
                max_row = data_plot.loc[idx_max]
                edad_max = int(pd.to_numeric(pd.Series([max_row["age"]]), errors="coerce").fillna(0).iloc[0])
                max_name = f"Max: {max_row['name']}"
                max_meta = f"{max_row['last_club_name']} | {edad_max}a"
                max_txt = f"{max_name}\n{max_meta}"
                color_max = str(max_row["color"])
                x_max = float(max_row[metrica])
                y_max = float(max_row["y_jitter"])
                x_min = float(data_plot[metrica].min())
                x_max_axis = float(data_plot[metrica].max())
                x_mid = x_min + (x_max_axis - x_min) / 2.0

                ax.set_title(nombre_metrica_legible(metrica), fontsize=9, pad=2)
                # Etiqueta del máximo dentro del gráfico y conectada a su bolita
                if x_max >= x_mid:
                    txt_offset = (-18, 22)
                    ha_txt = "right"
                    rad = 0.2
                else:
                    txt_offset = (18, 22)
                    ha_txt = "left"
                    rad = -0.2
                ax.annotate(
                    max_txt,
                    xy=(x_max, y_max),
                    xytext=txt_offset,
                    textcoords="offset points",
                    ha=ha_txt,
                    va="bottom",
                    fontsize=7.0,
                    color="#111111",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color_max, ec="#222222", lw=1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color_max,
                        lw=1.0,
                        connectionstyle=f"arc3,rad={rad}",
                    ),
                    annotation_clip=True,
                    zorder=7,
                )
                ax.set_yticks([])
                ax.grid(axis="x", alpha=0.25, linestyle="--")
                ax.tick_params(axis="x", labelsize=8.5)
                ax.set_xlabel("")
                ax.set_ylim(-0.35, 0.35)

            for j in range(min(6, len(metricas)), 6):
                axes[j].axis("off")

            comp_txt = f" | Comparado con: {jugador_b.get('name')}" if jugador_b is not None else ""
            fig.suptitle(
                f"Swarmplots {categoria} | Q1-Q4 (rojo->verde) | Base: {jugador_a.get('name')}{comp_txt}",
                fontsize=10,
                y=0.998,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.985])
            return fig

        for cat_name, metricas_cat in categorias.items():
            metricas_ok = [m for m in metricas_cat if m in df_all.columns]
            with col_map[cat_name]:
                if not metricas_ok:
                    st.warning(f"No hay métricas para {cat_name}.")
                    continue

                vals_base = percentiles_jugador(df_ref, row_base, metricas_ok)
                if row_cmp is None:
                    vals_cmp = percentiles_promedio(df_ref, metricas_ok)
                else:
                    vals_cmp = percentiles_jugador(df_ref, row_cmp, metricas_ok)

                titulo = f"{cat_name} | {referencia_txt}"
                fig = crear_radar_pizza(metricas_ok, vals_base, vals_cmp, info_base, cmp_label, titulo)
                fig_swarm = crear_swarmplots_categoria(
                    metricas=metricas_ok,
                    categoria=cat_name,
                    jugador_a=row_base,
                    jugador_b=row_cmp,
                )
                st.pyplot(fig, use_container_width=False)
                st.pyplot(fig_swarm, use_container_width=True)
                plt.close(fig)
                plt.close(fig_swarm)


    if df_radar.empty:
        st.warning("No hay datos disponibles con los filtros seleccionados para el radar.")
    else:
        df_radar = df_radar.copy()
        df_radar["pos_general_4"] = df_radar["position_gen_ESP1"].apply(posicion_general_4)
        df_radar["jugador_label"] = (
            df_radar["name"].astype(str) + " (" + df_radar["last_club_name"].astype(str) + ")"
        )

        metricas_base = sorted(
            [
                c
                for c in df_radar.select_dtypes(include="number").columns
                if c not in {"id", "Group", "age", "minutes_on_field", "pct_min_jugados"}
            ]
        )

        metricas_std = {
            "Ofensivo": [
                "non_penalty_goal_avg",
                "xg_shot_avg",
                "assists_avg",
                "xg_assist_avg",
                "offensive_duels_won",
                "successful_attacking_actions_avg",
            ],
            "Defensivo": [
                "defensive_duels_won",
                "interceptions_avg",
                "successful_defensive_actions_avg",
                "aerial_duels_won",
                "shot_block_avg",
                "fouls_avg",
            ],
            "Control": [
                "passes_avg",
                "accurate_passes_percent",
                "progressive_pass_avg",
                "passes_to_final_third_avg",
                "successful_forward_passes_percent",
                "successful_long_passes_percent",
            ],
        }

        st.markdown("### Métricas por radar (editable, máximo 6 por categoría)")
        mcol1, mcol2, mcol3 = st.columns(3)
        categorias_config = {}
        for col, categoria in zip([mcol1, mcol2, mcol3], ["Ofensivo", "Defensivo", "Control"]):
            default_vals = [m for m in metricas_std[categoria] if m in metricas_base][:6]
            with col:
                categorias_config[categoria] = st.multiselect(
                    f"Métricas {categoria}",
                    metricas_base,
                    default=default_vals,
                    max_selections=6,
                    key=f"radar_metricas_{categoria.lower()}",
                )

        jugador_sel = st.selectbox(
            "Jugador base:",
            sorted(df_radar["jugador_label"].dropna().unique().tolist()),
            key="radar_jugador_base",
        )
        jugador_row = df_radar[df_radar["jugador_label"] == jugador_sel].iloc[0]

        tipo_referencia = st.radio(
            "Referencia de percentiles:",
            ["Posición específica", "Posición general (Portero/Defensa/Centrocampista/Delantero)"],
            horizontal=False,
            key="radar_tipo_ref",
        )

        if tipo_referencia == "Posición específica":
            pos_val = jugador_row["position_gen_ESP1"]
            df_ref = df_radar[df_radar["position_gen_ESP1"] == pos_val].copy()
            etiqueta_ref = f"Promedio posición específica ({pos_val})"
        else:
            pos_val = jugador_row["pos_general_4"]
            df_ref = df_radar[df_radar["pos_general_4"] == pos_val].copy()
            etiqueta_ref = f"Promedio posición general ({pos_val})"

        if df_ref.empty:
            df_ref = df_radar.copy()
            etiqueta_ref = "Promedio global (fallback)"

        st.caption(f"Referencia usada para percentiles: {etiqueta_ref} | Jugadores en referencia: {len(df_ref)}")

        modo_comparacion = st.radio(
            "Modo radar:",
            ["Jugador vs promedio de la referencia", "Jugador vs hasta 2 jugadores"],
            key="radar_modo_comparacion",
        )

        if modo_comparacion == "Jugador vs promedio de la referencia":
            render_three_radars(
                df_all=df_radar,
                df_ref=df_ref,
                row_base=jugador_row,
                row_cmp=None,
                cmp_label=f"{etiqueta_ref} | n={len(df_ref)}",
                referencia_txt=f"Comparativa vs {etiqueta_ref}",
                categorias=categorias_config,
            )
        else:
            opciones_cmp = [j for j in sorted(df_radar["jugador_label"].unique().tolist()) if j != jugador_sel]
            jugadores_cmp = st.multiselect(
                "Selecciona 1 o 2 jugadores a comparar:",
                opciones_cmp,
                max_selections=2,
                key="radar_jugadores_cmp",
            )

            if len(jugadores_cmp) == 0:
                st.info("Selecciona al menos un jugador para comparar.")
            else:
                if len(jugadores_cmp) == 2:
                    st.markdown(
                        f"**Comparadores seleccionados**  \n"
                        f"Jugador 1: `{jugadores_cmp[0]}`  \n"
                        f"Jugador 2: `{jugadores_cmp[1]}`"
                    )
                for i, jugador_cmp in enumerate(jugadores_cmp, start=1):
                    cmp_row = df_radar[df_radar["jugador_label"] == jugador_cmp].iloc[0]
                    st.markdown(f"### Comparador {i}: {jugador_cmp}")
                    render_three_radars(
                        df_all=df_radar,
                        df_ref=df_ref,
                        row_base=jugador_row,
                        row_cmp=cmp_row,
                        cmp_label=player_info_label(cmp_row),
                        referencia_txt=f"Comparativa directa ({etiqueta_ref})",
                        categorias=categorias_config,
                    )
