import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, percentileofscore
from mplsoccer import PyPizza, FontManager
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import numpy as np
import io
import os
import streamlit as st
from common.reporte_fpdf import PDF, PDF_OUTPUT_DIR
import re
import unidecode
from common.reporte_fpdf import generar_pagina_cierre_con_fondo
from common.data_loader import load_players_data
from common.app_config import is_home_only_mode
from common.sidebar_branding import render_sidebar_branding
from common.theme import apply_app_theme

if is_home_only_mode():
    st.warning("Esta página está deshabilitada en modo alumnos.")
    st.stop()

render_sidebar_branding()
apply_app_theme()

def safe_filename(name):
    name = unidecode.unidecode(name)
    name = re.sub(r"[^\w\s-]", "", name)
    name = name.replace(" ", "_")
    return name


# === Rutas de carpetas para figuras del PDF ===
FIG_TEMP_DIR = "figures"
PIZZA_DIR = os.path.join(FIG_TEMP_DIR, "pizzas")

# ==================== DEFINICIÓN DE MÉTRICAS POR CATEGORÍA ====================

METRICAS_CATEGORIAS = {
    'Ofensiva': [
        'non_penalty_goal_avg',
        'head_goals_avg', 
        'xg_shot_avg',
        'shots_on_target_percent',
        'assists_avg',
        'xg_assist_avg',
        'offensive_duels_avg',
        'offensive_duels_won',
        'successful_attacking_actions_avg',
        'successful_dribbles_percent',
        'accurate_crosses_percent',
        'foul_suffered_avg'
    ],
    'Defensiva': [
        'defensive_duels_avg',
        'defensive_duels_won',
        'successful_defensive_actions_avg',
        'aerial_duels_avg',
        'aerial_duels_won',
        'interceptions_avg',
        'tackle_avg',
        'possession_adjusted_interceptions',
        'possession_adjusted_tackle',
        'fouls_avg'
    ],
    'Control': [
        'passes_avg',
        'accurate_passes_percent',
        'deep_completed_pass_avg',
        'progressive_pass_avg',
        'passes_to_final_third_avg',
        'accurate_passes_to_final_third_percent',
        'forward_passes_avg',
        'successful_forward_passes_percent',
        'long_passes_avg',
        'successful_long_passes_percent'
    ]
}

# ==================== FUNCIONES AUXILIARES ====================

def calcular_kpis_para_un_jugador(jugador_row, df_referencia, metricas_categorias, excluir_jugador=True):
    """
    Calcula los KPIs de Ataque, Defensa, Control y Total para un jugador específico
    basado en percentiles respecto a un dataframe de referencia.
    
    Args:
        jugador_row: Serie/dict con los datos del jugador
        df_referencia: DataFrame de referencia para calcular percentiles
        metricas_categorias: Dict con las métricas por categoría
        excluir_jugador: Si True, excluye al jugador de la muestra de referencia
    
    Returns:
        tuple: (ataque, defensa, control, total)
    """
    from scipy.stats import percentileofscore
    
    # Si debemos excluir al jugador, crear una muestra sin él
    if excluir_jugador and "jugador_id" in jugador_row:
        jugador_id = jugador_row["jugador_id"]
        df_ref = df_referencia[df_referencia["jugador_id"] != jugador_id].copy()
    else:
        df_ref = df_referencia.copy()
    
    kpis = {}
    
    # Mapear nuestras categorías a los nombres que queremos
    mapeo_categorias = {
        'Ofensiva': 'Ataque',
        'Defensiva': 'Defensa', 
        'Control': 'Control'
    }
    
    for categoria_original, categoria_kpi in mapeo_categorias.items():
        if categoria_original not in metricas_categorias:
            continue
            
        metricas = metricas_categorias[categoria_original]
        percentiles = []
        
        for metrica in metricas:
            if metrica in df_ref.columns:
                try:
                    # Obtener valor del jugador
                    valor_jugador = jugador_row.get(metrica, 0)
                    if pd.isna(valor_jugador):
                        valor_jugador = 0
                    
                    # Calcular percentil respecto al dataframe de referencia (SIN el jugador)
                    valores_referencia = df_ref[metrica].fillna(0)
                    percentil = percentileofscore(valores_referencia, valor_jugador, kind='rank')
                    percentiles.append(percentil)
                except Exception as e:
                    # Si hay error, usar percentil 0
                    percentiles.append(0)
        
        # Promedio de percentiles para esta categoría
        kpis[categoria_kpi] = np.mean(percentiles) if percentiles else 0.0
    
    # Extraer valores y redondear
    ataque = round(kpis.get("Ataque", 0), 2)
    defensa = round(kpis.get("Defensa", 0), 2) 
    control = round(kpis.get("Control", 0), 2)
    total = round(np.mean([ataque, defensa, control]), 2)
    
    return ataque, defensa, control, total

def color_for_kpi(value):
    """Devuelve el color RGB para un valor de KPI"""
    if value >= 75:
        return 168, 230, 168  # verde
    elif value >= 60:
        return 255, 252, 151  # amarillo
    elif value >= 45:
        return 255, 182, 142  # naranja claro
    elif value >= 30:
        return 231, 154, 154  # naranja oscuro
    else:
        return 229, 115, 115  # rojo

def crear_kpi_cards_visual(ataque, defensa, control, total):
    """Crea una visualización de las tarjetas KPI usando matplotlib"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    etiquetas = ["Ataque", "Defensa", "Control", "Total"]
    valores = [ataque, defensa, control, total]
    
    for i, (ax, etiqueta, valor) in enumerate(zip(axes, etiquetas, valores)):
        # Color según valor
        r, g, b = color_for_kpi(valor)
        color_normalizado = (r/255, g/255, b/255)
        
        # Crear tarjeta
        ax.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.4, 
                                 facecolor=color_normalizado, 
                                 edgecolor='black', 
                                 linewidth=2))
        
        # Texto del valor
        ax.text(0.5, 0.5, f"{valor:.1f}", 
               ha='center', va='center', 
               fontsize=16, fontweight='bold')
        
        # Etiqueta
        ax.text(0.5, 0.15, etiqueta, 
               ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        # Configurar ejes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle("Indicadores de rendimiento (percentiles)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def calcular_zscore_jugador_vs_grupo(df_jugadores_seleccionados, jugador_data, metricas_sel):
    """Calcula el Z-score del jugador respecto SOLO a los jugadores seleccionados."""
    from scipy.stats import norm
    
    if len(df_jugadores_seleccionados) < 2:
        return [50] * len(metricas_sel), [0] * len(metricas_sel)
    
    zscores = []
    percentiles_zscore = []
    
    for metrica in metricas_sel:
        valores_grupo = df_jugadores_seleccionados[metrica].dropna()
        valor_jugador = jugador_data.get(metrica, 0)
        
        if pd.isna(valor_jugador) or len(valores_grupo) < 2:
            zscores.append(0)
            percentiles_zscore.append(50)
            continue
        
        media_grupo = valores_grupo.mean()
        std_grupo = valores_grupo.std()
        
        if std_grupo == 0:
            zscores.append(0)
            percentiles_zscore.append(50)
            continue
        
        zscore = (valor_jugador - media_grupo) / std_grupo
        percentil_zscore = max(1, min(99, norm.cdf(zscore) * 100))
        
        zscores.append(zscore)
        percentiles_zscore.append(int(percentil_zscore))
    
    return percentiles_zscore, zscores

def interpretar_zscore(zscore):
    """Interpreta el valor del Z-score de forma comprensible."""
    if zscore > 2:
        return "🟢 Excelente (>2σ)"
    elif zscore > 1:
        return "🔵 Muy bueno (1-2σ)"
    elif zscore > 0.5:
        return "🟡 Bueno (0.5-1σ)"
    elif zscore >= -0.5:
        return "🟠 Promedio (±0.5σ)"
    elif zscore >= -1:
        return "🔴 Por debajo (-0.5 a -1σ)"
    else:
        return "⚫ Muy por debajo (<-1σ)"

def crear_leyenda_radar_categoria(categoria):
    """Crea la leyenda específica para cada categoría de radar chart"""
    
    leyendas = {
        'Ofensiva': {
            'titulo': '🎯 LEYENDA - RADAR OFENSIVO',
            'color_principal': '#FF5733',
            'color_secundario': '#4CAF50',
            'descripcion_principal': 'Percentil Global',
            'descripcion_secundaria': 'Z-score Grupal',
            'explicacion': """
            **🔴 Área Roja (Exterior)**: Percentil global del jugador comparado con toda la muestra filtrada
            **🟢 Área Verde (Interior)**: Z-score del jugador comparado con el grupo seleccionado
            
            **📊 Interpretación Ofensiva:**
            - **Valores altos**: Mayor capacidad goleadora, creatividad y aporte ofensivo
            - **Área roja > verde**: El jugador está mejor posicionado globalmente que en el grupo específico
            - **Área verde > roja**: El jugador destaca especialmente dentro del grupo seleccionado
            
            **🎯 Métricas clave incluidas:**
            - Goles y xG por partido
            - Asistencias y xA por partido  
            - Precisión de tiros a portería
            - Duelos ofensivos ganados
            - Regates exitosos
            - Centros precisos
            """
        },
        
        'Defensiva': {
            'titulo': '🛡️ LEYENDA - RADAR DEFENSIVO',
            'color_principal': '#2E7D32',
            'color_secundario': '#FFA726',
            'descripcion_principal': 'Percentil Global',
            'descripcion_secundaria': 'Z-score Grupal',
            'explicacion': """
            **🟢 Área Verde Oscuro (Exterior)**: Percentil global del jugador comparado con toda la muestra filtrada
            **🟠 Área Naranja (Interior)**: Z-score del jugador comparado con el grupo seleccionado
            
            **📊 Interpretación Defensiva:**
            - **Valores altos**: Mayor solidez defensiva, recuperación de balón y duelos aéreos
            - **Área verde > naranja**: El jugador está mejor posicionado globalmente que en el grupo específico
            - **Área naranja > verde**: El jugador destaca especialmente dentro del grupo seleccionado
            
            **🛡️ Métricas clave incluidas:**
            - Duelos defensivos ganados
            - Intercepciones por partido
            - Entradas exitosas
            - Duelos aéreos ganados
            - Acciones defensivas exitosas
            - Entradas ajustadas por posesión
            """
        },
        
        'Control': {
            'titulo': '⚽ LEYENDA - RADAR CONTROL',
            'color_principal': '#2196F3',
            'color_secundario': '#4CAF50',
            'descripcion_principal': 'Percentil Global',
            'descripcion_secundaria': 'Z-score Grupal',
            'explicacion': """
            **🔵 Área Azul (Exterior)**: Percentil global del jugador comparado con toda la muestra filtrada
            **🟢 Área Verde (Interior)**: Z-score del jugador comparado con el grupo seleccionado
            
            **📊 Interpretación Control:**
            - **Valores altos**: Mayor dominio del balón, precisión de pase y construcción de juego
            - **Área azul > verde**: El jugador está mejor posicionado globalmente que en el grupo específico
            - **Área verde > azul**: El jugador destaca especialmente dentro del grupo seleccionado
            
            **⚽ Métricas clave incluidas:**
            - Pases por partido y precisión
            - Pases progresivos
            - Pases al último tercio
            - Pases hacia adelante exitosos
            - Pases largos precisos
            - Pases profundos completados
            """
        }
    }
    
    return leyendas.get(categoria, {})


def crear_pizza_chart_categoria(metricas, valores_jugador, valores_promedio, jugador, categoria, color_principal):
    """Crea un pizza chart para una categoría específica - VERSIÓN CORREGIDA"""
    try:
        # Colores más diferenciados para cada categoría
        if categoria == "Defensiva":
            color_principal = "#2E7D32"  # Verde más oscuro y diferente
            color_comparacion = "#FFA726"  # Naranja para mayor contraste
        else:
            color_comparacion = "#4CAF50"  # Verde estándar para otras categorías

        # CORRECCIÓN DEL REDONDEADO: Validar valores antes de redondear
        valores_jugador_clean = []
        valores_promedio_clean = []
        
        for v in valores_jugador:
            if pd.isna(v) or v is None:
                valores_jugador_clean.append(0.0)
            else:
                # Asegurar que esté en rango 0-100 para percentiles
                val_clean = max(0, min(100, float(v)))
                valores_jugador_clean.append(round(val_clean, 1))  # 1 decimal es suficiente
        
        for v in valores_promedio:
            if pd.isna(v) or v is None:
                valores_promedio_clean.append(0.0)
            else:
                # Asegurar que esté en rango 0-100 para percentiles
                val_clean = max(0, min(100, float(v)))
                valores_promedio_clean.append(round(val_clean, 1))  # 1 decimal es suficiente
        
        # Acortar nombres de métricas para mejor visualización
        metricas_cortas = []
        for metrica in metricas:
            # Remover palabras comunes y acortar
            metrica_corta = metrica.replace("Avg", "").replace("Percent", "%")
            metrica_corta = metrica_corta.replace("Successful", "Succ.")
            metrica_corta = metrica_corta.replace("Average", "Avg")
            metrica_corta = metrica_corta.replace("Actions", "Act.")
            metrica_corta = metrica_corta.replace("Possession Adjusted", "Poss.Adj.")
            
            # Si es muy largo, tomar primeras palabras
            words = metrica_corta.split()
            if len(words) > 2:
                metrica_corta = " ".join(words[:2])
            
            # Límite de caracteres
            if len(metrica_corta) > 15:
                metrica_corta = metrica_corta[:12] + "..."
                
            metricas_cortas.append(metrica_corta)
        
        baker = PyPizza(
            params=metricas_cortas,
            background_color="#ffffff",
            straight_line_color="#000000",
            straight_line_lw=1,
            last_circle_lw=1,
            last_circle_color="#000000",
            other_circle_ls="-.",
            other_circle_lw=0.8
        )
        
        fig, ax = baker.make_pizza(
            valores_jugador_clean,
            compare_values=valores_promedio_clean,
            figsize=(7, 7),
            kwargs_slices=dict(
                facecolor=color_principal, edgecolor="#222222",
                zorder=2, linewidth=2, alpha=0.7
            ),
            kwargs_compare=dict(
                facecolor=color_comparacion, edgecolor="#222222",
                zorder=1, linewidth=2, alpha=0.7
            ),
            kwargs_params=dict(
                color="#000000", fontsize=9, va="center", fontweight="bold"
            ),
            kwargs_values=dict(
                color="#000000", fontsize=7, zorder=3, fontweight="bold",
                bbox=dict(edgecolor="#000000", facecolor=color_principal, boxstyle="round,pad=0.1", lw=1)
            ),
            kwargs_compare_values=dict(
                color="#000000", fontsize=7, zorder=3, fontweight="bold",
                bbox=dict(edgecolor="#000000", facecolor=color_comparacion, boxstyle="round,pad=0.1", lw=1)
            )
        )
        
        # Título de la categoría
        fig.text(
            0.5, 0.95, 
            f"{categoria.upper()}",
            size=16, ha="center", color=color_principal, fontweight="bold"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error al crear pizza chart {categoria}: {str(e)}")
        return None

@st.cache_data
def cargar_datos():
    """Carga y preprocesa los datos de jugadores"""
    try:
        df = load_players_data()
        
        # Limpieza y conversión de datos
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0)
        df["minutes_on_field"] = pd.to_numeric(df["minutes_on_field"], errors="coerce").fillna(0)
        
        # Crear identificador único de jugador
        df["jugador_id"] = df["name"].astype(str) + " (" + df["last_club_name"].astype(str) + ")"
        
        # Limpiar valores nulos en columnas clave
        df["League"] = df["League"].fillna("Liga Desconocida")
        df["primary_position_ESP"] = df["primary_position_ESP"].fillna("Posición Desconocida")
        
        return df
    except FileNotFoundError:
        st.error("❌ No se pudo encontrar el archivo CSV configurado")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

def normalizar_datos(df, metricas):
    """Normaliza los datos usando Min-Max scaling"""
    df_norm = df.copy()
    
    for metrica in metricas:
        if metrica in df_norm.columns:
            minimo = df_norm[metrica].min()
            maximo = df_norm[metrica].max()
            
            if maximo - minimo > 0:
                df_norm[metrica] = (df_norm[metrica] - minimo) / (maximo - minimo)
            else:
                df_norm[metrica] = 0.5
    
    return df_norm

def calcular_scoring(df_norm, metricas):
    """Calcula el scoring general basado en las métricas seleccionadas"""
    if not metricas:
        return df_norm
    
    # Solo usar métricas que existen en el dataframe
    metricas_disponibles = [m for m in metricas if m in df_norm.columns]
    
    if not metricas_disponibles:
        df_norm["Scoring"] = 5.5
        return df_norm
    
    df_norm["Scoring"] = df_norm[metricas_disponibles].sum(axis=1)
    
    # Normalizar scoring a escala 1-10
    min_s, max_s = df_norm["Scoring"].min(), df_norm["Scoring"].max()
    if max_s - min_s > 0:
        df_norm["Scoring"] = 1 + (df_norm["Scoring"] - min_s) * 9 / (max_s - min_s)
    else:
        df_norm["Scoring"] = 5.5
    
    return df_norm

def crear_tabla_comparativa_categoria(df_norm, metricas_sel, categoria, color_categoria):
    """Crea una tabla comparativa para una categoría específica - VERSIÓN ORIGINAL COMPACTA"""
    df_display = df_norm.sort_values("Scoring", ascending=False).reset_index(drop=True)
    df_display["Position"] = df_display.index + 1
    
    # Solo incluir métricas que existen en el dataframe (TODAS, sin cortar)
    metricas_disponibles = [m for m in metricas_sel if m in df_display.columns]
    columnas_finales = ["Position", "Scoring"] + metricas_disponibles
    
    if not metricas_disponibles:
        st.warning(f"⚠️ No hay métricas disponibles para la categoría {categoria}")
        return None
    
    # VOLVER AL TAMAÑO ORIGINAL QUE FUNCIONABA
    fig, ax = plt.subplots(figsize=(max(len(columnas_finales) * 1.2, 8), len(df_display) * 0.6))
    ax.axis("tight")
    ax.axis("off")
    
    # Crear colormap suave
    cmap = LinearSegmentedColormap.from_list("categoria_colors", [
        "#FFB3B3",  # Rojo suave para valores bajos
        "#FFD4AA",  # Naranja suave para valores medios  
        "#B3D9B3"   # Verde suave para valores altos
    ])
    
    # Crear tabla con datos COMO ANTES (2 decimales)
    table_data = df_display[columnas_finales].round(2).values
    
    # HEADERS LEGIBLES Y COMPACTOS (sin cortar métricas)
    headers = []
    for col in columnas_finales:
        if col == "Position":
            headers.append("Pos")
        elif col == "Scoring":
            headers.append("Score")
        else:
            # ABREVIACIONES INTELIGENTES PERO LEGIBLES
            header = col.replace("_", " ").title()
            header = header.replace("Avg", "")
            header = header.replace("Percent", "%")
            header = header.replace("Successful", "Succ")
            header = header.replace("Accurate", "Acc")
            header = header.replace("Offensive", "Off")
            header = header.replace("Defensive", "Def")
            header = header.replace("Actions", "Act")
            header = header.replace("Duels", "Du")
            header = header.replace("Won", "W")
            header = header.replace("Possession Adjusted", "Poss Adj")
            header = header.replace("To Final Third", "Final 3rd")
            header = header.replace("Forward Passes", "Fwd Pass")
            header = header.replace("Long Passes", "Long Pass")
            header = header.replace("Deep Completed Pass", "Deep Pass")
            header = header.replace("Progressive Pass", "Prog Pass")
            header = header.replace("Passes To Final Third", "Pass Final 3rd")
            header = header.replace("Acc Pass To Final Third", "Acc Final 3rd")
            header = header.replace("Succ Forward Pass", "Succ Fwd Pass")
            header = header.replace("Succ Long Pass", "Succ Long Pass")
            
            # Si aún es muy largo, acortar más
            if len(header) > 12:
                words = header.split()
                if len(words) > 2:
                    header = f"{words[0]} {words[1][:4]}"
                    
            headers.append(header)
    
    # Row labels CON EQUIPO ENTRE PARÉNTESIS (manteniendo formato original)
    row_labels = df_display["jugador_id"].tolist()
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center"
    )
    
    # Aplicar colores (COMO ANTES)
    for i in range(len(df_display)):
        for j, col in enumerate(columnas_finales):
            val = df_display.iloc[i][col]
            cell = table[(i + 1, j)]
            
            if isinstance(val, (int, float)):
                if col == "Position":
                    cell.set_facecolor("#E8E8E8")
                    cell.set_text_props(fontweight="bold")
                elif col == "Scoring":
                    normalized_score = (val - 1) / 9
                    if normalized_score >= 0.95:
                        cell.set_facecolor("white")
                        cell.set_text_props(fontweight="bold")
                    else:
                        cell.set_facecolor(cmap(normalized_score))
                else:
                    if val >= 0.98:
                        cell.set_facecolor("white")
                        cell.set_text_props(fontweight="bold")
                    else:
                        cell.set_facecolor(cmap(val))
            
            # FUENTES PEQUEÑAS COMO ANTES
            cell.set_text_props(fontsize=8)
    
    # Formatear headers COMO ANTES
    for j in range(len(columnas_finales)):
        header = table[(0, j)]
        header.set_text_props(fontweight="bold", fontsize=9)
        header.set_facecolor(color_categoria)
        header.set_text_props(color="white")
    
    # Formatear row labels COMO ANTES
    for i in range(1, len(df_display) + 1):
        row_label = table[(i, -1)]
        row_label.set_text_props(fontweight="bold", fontsize=7)
        row_label.set_facecolor("#F0F0F0")
    
    table.auto_set_font_size(False)
    # ESCALA ORIGINAL QUE FUNCIONABA
    table.scale(1.2, 1.5)
    
    plt.title(f"📊 Ranking {categoria}", fontsize=14, fontweight="bold", color=color_categoria, pad=15)
    
    return fig

def crear_tabla_rankings_consolidada(df_sel, METRICAS_CATEGORIAS, COLORES_CATEGORIAS):
    """Crea una tabla consolidada con las posiciones de cada jugador en los 3 rankings - MISMO TAMAÑO"""
    
    # Diccionario para almacenar las posiciones de cada jugador en cada categoría
    posiciones_jugadores = {}
    
    # Calcular posición en cada categoría
    for categoria, metricas in METRICAS_CATEGORIAS.items():
        metricas_disponibles = [m for m in metricas if m in df_sel.columns]
        
        if not metricas_disponibles:
            continue
            
        # Crear ranking para esta categoría
        df_categoria = df_sel[["jugador_id"] + metricas_disponibles].copy()
        df_norm_cat = normalizar_datos(df_categoria, metricas_disponibles)
        df_norm_cat = calcular_scoring(df_norm_cat, metricas_disponibles)
        df_ranking = df_norm_cat.sort_values("Scoring", ascending=False).reset_index(drop=True)
        df_ranking["Position"] = df_ranking.index + 1
        
        # Guardar posiciones
        for _, row in df_ranking.iterrows():
            jugador_id = row["jugador_id"]
            if jugador_id not in posiciones_jugadores:
                posiciones_jugadores[jugador_id] = {}
            posiciones_jugadores[jugador_id][categoria] = int(row["Position"])
            posiciones_jugadores[jugador_id]["Scoring_" + categoria] = round(row["Scoring"], 1)
    
    # Crear DataFrame consolidado
    data_consolidada = []
    for jugador_id, posiciones in posiciones_jugadores.items():
        fila = {
            "Jugador": jugador_id,
            "RK_Ofensiva": posiciones.get("Ofensiva", "-"),
            "Score_Off": posiciones.get("Scoring_Ofensiva", "-"),
            "RK_Defensiva": posiciones.get("Defensiva", "-"), 
            "Score_Def": posiciones.get("Scoring_Defensiva", "-"),
            "RK_Control": posiciones.get("Control", "-"),
            "Score_Ctrl": posiciones.get("Scoring_Control", "-")
        }
        
        # Calcular promedio de posiciones (solo las que existen)
        posiciones_numericas = [v for k, v in posiciones.items() if not k.startswith("Scoring_") and isinstance(v, int)]
        if posiciones_numericas:
            fila["Promedio_RK"] = round(sum(posiciones_numericas) / len(posiciones_numericas), 1)
        else:
            fila["Promedio_RK"] = "-"
            
        data_consolidada.append(fila)
    
    # Convertir a DataFrame y ordenar por promedio
    df_consolidado = pd.DataFrame(data_consolidada)
    if "Promedio_RK" in df_consolidado.columns:
        # Ordenar por promedio (menor promedio = mejor)
        df_consolidado = df_consolidado.sort_values("Promedio_RK").reset_index(drop=True)
    
    # ✅ USAR EL MISMO TAMAÑO QUE LAS OTRAS TABLAS - SIN COLUMNA JUGADOR DUPLICADA
    columnas = ["RK_Ofensiva", "Score_Off", "RK_Defensiva", "Score_Def", "RK_Control", "Score_Ctrl", "Promedio_RK"]
    
    # CALCULAR TAMAÑO IGUAL QUE LAS OTRAS TABLAS
    fig, ax = plt.subplots(figsize=(max(len(columnas) * 1.2, 8), len(df_consolidado) * 0.6))
    ax.axis("tight")
    ax.axis("off")
    
    # Headers abreviados COMO LAS OTRAS TABLAS
    headers = ["RK Off", "Score Off", "RK Def", "Score Def", "RK Ctrl", "Score Ctrl", "Prom RK"]
    
    # Crear datos de la tabla SIN la columna Jugador (va en rowLabels)
    table_data = []
    for _, row in df_consolidado.iterrows():
        fila_tabla = []
        for col in columnas:
            val = row[col]
            fila_tabla.append(str(val))
        table_data.append(fila_tabla)
    
    # Row labels CON EQUIPO ENTRE PARÉNTESIS (igual que las otras)
    row_labels = df_consolidado["Jugador"].tolist()
    
    # Crear tabla
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center"
    )
    
    # CREAR GRADIENTE SUAVE DE VERDE (1º) A ROJO (ÚLTIMO)
    import matplotlib.colors as mcolors
    num_jugadores = len(df_consolidado)
    
    # Aplicar colores con gradiente suave
    for i in range(len(df_consolidado)):
        for j, col in enumerate(columnas):
            cell = table[(i + 1, j)]
            
            # Colores según el tipo de columna
            if "RK_" in col:
                # GRADIENTE SUAVE PARA RANKINGS
                try:
                    pos = int(df_consolidado.iloc[i][col])
                    # Calcular posición normalizada (0 = mejor, 1 = peor)
                    norm_pos = (pos - 1) / max(1, num_jugadores - 1)
                    
                    # Gradiente de verde claro a rojo claro
                    if norm_pos <= 0.33:  # Top 33% - Verde
                        color = mcolors.to_hex((0.7 + norm_pos * 0.3, 0.9, 0.7 + norm_pos * 0.2))
                    elif norm_pos <= 0.66:  # Middle 33% - Amarillo
                        color = mcolors.to_hex((0.95, 0.9 - (norm_pos - 0.33) * 0.2, 0.6))
                    else:  # Bottom 33% - Rojo
                        color = mcolors.to_hex((0.95, 0.7 - (norm_pos - 0.66) * 0.3, 0.6 - (norm_pos - 0.66) * 0.3))
                    
                    cell.set_facecolor(color)
                    if pos == 1:
                        cell.set_text_props(fontweight="bold")
                except:
                    cell.set_facecolor("#F5F5F5")
                    
            elif "Score_" in col:
                # Azul muy suave para scores
                cell.set_facecolor("#F0F8FF")
                
            elif col == "Promedio_RK":
                # GRADIENTE ESPECIAL PARA PROMEDIO
                try:
                    prom = float(df_consolidado.iloc[i][col])
                    # Calcular posición normalizada basada en el promedio
                    norm_prom = min(1.0, (prom - 1) / max(1, num_jugadores - 1))
                    
                    if norm_prom <= 0.33:  # Mejor tercio - Verde más intenso
                        color = mcolors.to_hex((0.6, 0.9, 0.6))
                        cell.set_text_props(fontweight="bold")
                    elif norm_prom <= 0.66:  # Tercio medio - Amarillo suave
                        color = mcolors.to_hex((0.95, 0.9, 0.6))
                    else:  # Peor tercio - Rojo suave
                        color = mcolors.to_hex((0.95, 0.7, 0.6))
                    
                    cell.set_facecolor(color)
                except:
                    cell.set_facecolor("#F5F5F5")
            
            # FUENTES IGUALES QUE LAS OTRAS TABLAS
            cell.set_text_props(fontsize=8)
    
    # Formatear headers IGUAL QUE LAS OTRAS TABLAS
    for j in range(len(headers)):
        header = table[(0, j)]
        header.set_text_props(fontweight="bold", fontsize=9)
        header.set_facecolor("#333333")  # Color gris para unificar
        header.set_text_props(color="white")
    
    # Formatear row labels IGUAL QUE LAS OTRAS TABLAS
    for i in range(1, len(df_consolidado) + 1):
        row_label = table[(i, -1)]
        row_label.set_text_props(fontweight="bold", fontsize=7)
        row_label.set_facecolor("#F0F0F0")
    
    table.auto_set_font_size(False)
    # ESCALA IGUAL QUE LAS OTRAS TABLAS
    table.scale(1.2, 1.5)
    
    plt.title("📊 Ranking Consolidado", fontsize=14, fontweight="bold", color="#333333", pad=15)
    
    return fig


def crear_scatterplot_interactivo(df_muestra, df_seleccionados, x_metric, y_metric, categoria, color_categoria):
    """Crea un scatterplot interactivo con Plotly - Sin leyenda y con textos ajustados"""
    try:
              
        fig = go.Figure()
        
        # Filtrar datos eliminando valores = 0
        if not df_muestra.empty and x_metric in df_muestra.columns and y_metric in df_muestra.columns:
            muestra_filtrada = df_muestra[
                (df_muestra[x_metric] != 0) & (df_muestra[y_metric] != 0) & 
                (pd.notna(df_muestra[x_metric])) & (pd.notna(df_muestra[y_metric]))
            ].copy()
            
            if not muestra_filtrada.empty:
                # IMPORTANTE: Ahora cada punto de la muestra tiene información de hover
                hover_texts = []
                for _, row in muestra_filtrada.iterrows():
                    # Extraer nombre y equipo del jugador_id
                    if "jugador_id" in row:
                        jugador_name = row["jugador_id"].split(" (")[0] if "(" in row["jugador_id"] else row["jugador_id"]
                        equipo_name = row["jugador_id"].split("(")[1].replace(")", "").strip() if "(" in row["jugador_id"] else "N/A"
                    else:
                        jugador_name = row.get("name", "Desconocido")
                        equipo_name = row.get("last_club_name", "N/A")
                    
                    # Crear texto de hover personalizado
                    hover_text = f"<b>{jugador_name}</b><br>"
                    hover_text += f"Equipo: {equipo_name}<br>"
                    hover_text += f"{x_metric.replace('_', ' ').title()}: {row[x_metric]:.2f}<br>"
                    hover_text += f"{y_metric.replace('_', ' ').title()}: {row[y_metric]:.2f}"
                    
                    # Agregar información adicional si está disponible
                    if "primary_position_ESP" in row:
                        hover_text += f"<br>Posición: {row.get('primary_position_ESP', 'N/A')}"
                    if "age" in row:
                        hover_text += f"<br>Edad: {row.get('age', 'N/A'):.0f}"
                    if "minutes_on_field" in row:
                        hover_text += f"<br>Minutos: {row.get('minutes_on_field', 0):.0f}"
                    
                    hover_texts.append(hover_text)
                
                # Agregar muestra comparativa CON HOVER para cada punto
                fig.add_trace(go.Scatter(
                    x=muestra_filtrada[x_metric],
                    y=muestra_filtrada[y_metric],
                    mode='markers',
                    marker=dict(
                        color='#CCCCCC',
                        size=8,
                        opacity=0.6,
                        line=dict(width=0.5, color='#999999')
                    ),
                    name='Otros jugadores',
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=11,
                        font_family="Arial"
                    ),
                    showlegend=False  # NO mostrar en leyenda
                ))
                
                # Calcular y agregar líneas promedio
                mean_x = muestra_filtrada[x_metric].mean()
                mean_y = muestra_filtrada[y_metric].mean()
                
                # Línea promedio horizontal
                fig.add_hline(y=mean_y, line_dash="dash", line_color="red", opacity=0.5,
                            annotation_text=f"Promedio Y: {mean_y:.2f}",
                            annotation_position="right")
                
                # Línea promedio vertical  
                fig.add_vline(x=mean_x, line_dash="dash", line_color="blue", opacity=0.5,
                            annotation_text=f"Promedio X: {mean_x:.2f}",
                            annotation_position="top")
        
        # Agregar jugadores seleccionados con colores diferenciados
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        jugadores_validos = 0
        
        # Listas para almacenar posiciones y nombres
        positions_x = []
        positions_y = []
        names = []
        colors_used = []
        hover_infos = []
        
        for j, (_, row) in enumerate(df_seleccionados.iterrows()):
            # Solo graficar si ambos valores no son 0 y no son NaN
            if (row[x_metric] != 0 and row[y_metric] != 0 and 
                pd.notna(row[x_metric]) and pd.notna(row[y_metric])):
                
                player_color = colors[j % len(colors)]
                
                # Extraer información del jugador
                if "jugador_id" in row:
                    jugador_name = row["jugador_id"].split(" (")[0]
                    equipo_name = row["jugador_id"].split("(")[1].replace(")", "").strip()
                else:
                    jugador_name = row.get("name", "Jugador")
                    equipo_name = row.get("last_club_name", "N/A")
                
                # Información detallada para el hover
                hover_info = f"<b>{jugador_name}</b> ⭐<br>"
                hover_info += f"<b>Equipo: {equipo_name}</b><br>"
                hover_info += f"<b>{x_metric.replace('_', ' ').title()}: {row[x_metric]:.2f}</b><br>"
                hover_info += f"<b>{y_metric.replace('_', ' ').title()}: {row[y_metric]:.2f}</b><br>"
                
                # Agregar información adicional
                if "primary_position_ESP" in row:
                    hover_info += f"Posición: {row.get('primary_position_ESP', 'N/A')}<br>"
                if "age" in row:
                    hover_info += f"Edad: {row.get('age', 'N/A'):.0f}<br>"
                if "minutes_on_field" in row:
                    hover_info += f"Minutos: {row.get('minutes_on_field', 0):.0f}"
                
                # Guardar posiciones y nombres para las anotaciones
                positions_x.append(row[x_metric])
                positions_y.append(row[y_metric])
                names.append(jugador_name)
                colors_used.append(player_color)
                hover_infos.append(hover_info)
                
                # Agregar el punto SIN texto (el texto irá como anotación)
                fig.add_trace(go.Scatter(
                    x=[row[x_metric]],
                    y=[row[y_metric]],
                    mode='markers',
                    marker=dict(
                        color=player_color,
                        size=14,
                        line=dict(color='black', width=2)
                    ),
                    name=jugador_name,
                    hovertemplate=hover_info + '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor=player_color,
                        font_size=12,
                        font_family="Arial",
                        font_color="white"
                    ),
                    showlegend=False  # NO mostrar en leyenda
                ))
                jugadores_validos += 1
        
        # Algoritmo para distribuir las etiquetas evitando solapamientos
        if positions_x and positions_y:
            # Calcular rangos para posicionamiento inteligente
            x_range = max(positions_x) - min(positions_x) if len(positions_x) > 1 else 1
            y_range = max(positions_y) - min(positions_y) if len(positions_y) > 1 else 1
            
            # Definir offsets en un patrón circular para evitar solapamientos
            angle_offsets = np.linspace(0, 2*np.pi, len(positions_x), endpoint=False)
            
            for i, (x, y, name, color) in enumerate(zip(positions_x, positions_y, names, colors_used)):
                # Calcular offset basado en la posición relativa del punto
                # y un patrón circular para distribuir las etiquetas
                angle = angle_offsets[i]
                
                # Ajustar la distancia del offset basado en la densidad de puntos
                offset_distance = 0.05  # Base offset como proporción del rango
                ax_offset = np.cos(angle) * offset_distance * x_range
                ay_offset = np.sin(angle) * offset_distance * y_range
                
                # Si hay muchos jugadores juntos, alternar las direcciones
                if i % 2 == 0:
                    ax_offset *= 1.5
                    ay_offset *= 1.5
                
                # Agregar anotación con flecha
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=f"<b>{name}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=color,
                    ax=ax_offset * 100,  # Offset en píxeles
                    ay=-ay_offset * 100 if ay_offset > 0 else ay_offset * 100,  # Invertir si es necesario
                    bgcolor=color,
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(
                        size=10,
                        color="white",
                        family="Arial Black"
                    ),
                    opacity=0.9
                )
        
        # Crear información del contexto comparativo
        if not df_muestra.empty:
            contexto_info = f"Contexto Comparativo: {len(df_muestra)} jugadores"
            if 'primary_position_ESP' in df_muestra.columns:
                posiciones = df_muestra['primary_position_ESP'].dropna().unique()
                if len(posiciones) <= 3:
                    contexto_info += f" | Posiciones: {', '.join(posiciones)}"
                else:
                    contexto_info += f" | {len(posiciones)} posiciones"
            
            if 'age' in df_muestra.columns:
                edad_min = int(df_muestra['age'].min())
                edad_max = int(df_muestra['age'].max())
                edad_promedio = df_muestra['age'].mean()
                contexto_info += f" | Edad: {edad_min}-{edad_max} años (prom. {edad_promedio:.1f})"
            
            if 'minutes_on_field' in df_muestra.columns:
                min_promedio = int(df_muestra['minutes_on_field'].mean())
                contexto_info += f" | Min. promedio: {min_promedio}"
        else:
            contexto_info = "Sin contexto comparativo disponible"

        # Configurar layout SIN leyenda
        # Información del contexto más compacta
        if not df_muestra.empty:
            n_jugadores = len(df_muestra)
            edad_prom = df_muestra['age'].mean() if 'age' in df_muestra.columns else 0
            min_prom = int(df_muestra['minutes_on_field'].mean()) if 'minutes_on_field' in df_muestra.columns else 0
            contexto_simple = f"{n_jugadores} jugadores | Edad prom: {edad_prom:.1f} | Min prom: {min_prom}"
        else:
            contexto_simple = "Sin muestra comparativa"

        # Configurar layout SIN leyenda
        fig.update_layout(
            title={
                'text': f"<b>{categoria}</b><br>{x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()}<br><span style='font-size:12px'>{jugadores_validos} seleccionados | {contexto_simple}</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': color_categoria}
            },
            xaxis_title=x_metric.replace("_", " ").title(),
            yaxis_title=y_metric.replace("_", " ").title(),
            hovermode='closest',
            showlegend=False,
            width=600,
            height=520,  # Más altura para el título
            plot_bgcolor='white',
            margin=dict(t=80)  # Más margen arriba para el título
        )
        
    except ImportError:
        st.error("📦 Plotly no está instalado. Instálalo con: pip install plotly")
        return None
    except Exception as e:
        st.error(f"Error creando scatterplot interactivo: {str(e)}")
        return None
    return fig
# ==================== CONFIGURACIÓN INICIAL ====================

st.set_page_config(
    page_title="Análisis por Categorías", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2e8b57;
        padding-left: 1rem;
    }
    .categoria-header {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .ofensiva { background-color: #ffebee; color: #c62828; }
    .defensiva { background-color: #e8f5e8; color: #2e7d32; }
    .control { background-color: #e3f2fd; color: #1565c0; }
</style>
""", unsafe_allow_html=True)

# ==================== INTERFAZ PRINCIPAL ====================

st.markdown('<div class="main-header">⚽ Análisis por Categorías: Ofensiva | Defensiva | Control</div>', unsafe_allow_html=True)

# Cargar datos
with st.spinner("🔄 Cargando datos..."):
    df = cargar_datos()

if df.empty:
    st.stop()

# ==================== SIDEBAR - FILTROS PRINCIPALES ====================
st.sidebar.markdown("## 🎛️ Filtros de Selección")

# Filtros básicos
ligas_disponibles = [""] + sorted(df["League"].unique())
posiciones_disponibles = [""] + sorted(df["primary_position_ESP"].dropna().unique())

ligas_sel = st.sidebar.multiselect("🏆 Liga(s):", ligas_disponibles)
pos_esp_sel = st.sidebar.multiselect("⚽ Posiciones:", posiciones_disponibles)

# Aplicar filtros
filtro = df.copy()
if ligas_sel:
    filtro = filtro[filtro["League"].isin(ligas_sel)]
if pos_esp_sel:
    filtro = filtro[filtro["primary_position_ESP"].isin(pos_esp_sel)]

# ==================== SELECCIÓN DE JUGADORES ====================
st.markdown('<div class="section-header">🎯 Selección de Jugadores</div>', unsafe_allow_html=True)

jugadores_disponibles = sorted(filtro["jugador_id"].dropna().unique())

if not jugadores_disponibles:
    st.warning("⚠️ No se encontraron jugadores con los filtros seleccionados")
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    seleccionados = st.multiselect(
        "Selecciona jugadores para comparar:",
        jugadores_disponibles,
        help="Selecciona jugadores para análisis por categorías"
    )

with col2:
    st.metric("Jugadores disponibles", len(jugadores_disponibles))
    st.metric("Jugadores seleccionados", len(seleccionados))

# ==================== ANÁLISIS POR CATEGORÍAS ====================
if seleccionados:
    df_sel = df[df["jugador_id"].isin(seleccionados)].copy()
    
    # Definir colores para cada categoría
    COLORES_CATEGORIAS = {
        'Ofensiva': '#FF5733',
        'Defensiva': '#4CAF50', 
        'Control': '#2196F3'
    }
    
    # ==================== TABLAS COMPARATIVAS POR CATEGORÍA ====================
    st.markdown('<div class="section-header">📊 Rankings por Categoría</div>', unsafe_allow_html=True)
    
    for categoria, metricas in METRICAS_CATEGORIAS.items():
        st.markdown(f'<div class="categoria-header {categoria.lower()}">{categoria.upper()} 🎯</div>', unsafe_allow_html=True)
        
        # Verificar qué métricas están disponibles en el dataset
        metricas_disponibles = [m for m in metricas if m in df_sel.columns]
        
        if not metricas_disponibles:
            st.warning(f"⚠️ No se encontraron métricas de {categoria} en el dataset")
            continue
            
        st.info(f"📈 Métricas disponibles para {categoria}: {len(metricas_disponibles)}/{len(metricas)}")
        
        # Normalizar datos y calcular scoring para esta categoría
        df_categoria = df_sel[["jugador_id"] + metricas_disponibles].copy()
        df_norm_cat = normalizar_datos(df_categoria, metricas_disponibles)
        df_norm_cat = calcular_scoring(df_norm_cat, metricas_disponibles)
        
        # Crear tabla comparativa
        fig_tabla = crear_tabla_comparativa_categoria(
            df_norm_cat, metricas_disponibles, categoria, COLORES_CATEGORIAS[categoria]
        )
        if fig_tabla:
            st.pyplot(fig_tabla)
        
        # Mostrar métricas disponibles vs faltantes
        metricas_faltantes = [m for m in metricas if m not in df_sel.columns]
        if metricas_faltantes:
            with st.expander(f"ℹ️ Métricas no encontradas en {categoria}"):
                st.write("Las siguientes métricas no están disponibles en el dataset:")
                for metrica in metricas_faltantes:
                    st.write(f"• {metrica}")

    # ==================== TABLA CONSOLIDADA DE RANKINGS ====================
    st.markdown("### 📋 Resumen General de Rankings")
    st.info("💡 Esta tabla muestra la posición de cada jugador en los 3 rankings y su promedio general")
    
    fig_consolidada = crear_tabla_rankings_consolidada(df_sel, METRICAS_CATEGORIAS, COLORES_CATEGORIAS)
    if fig_consolidada:
        st.pyplot(fig_consolidada)
        
    with st.expander("📖 Interpretación de la Tabla de Rankings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #1976d2;">', unsafe_allow_html=True)
            st.markdown("### 🏆 <span style='color: #1976d2'>Columnas de la tabla</span>", unsafe_allow_html=True)
            st.markdown("""
            - **RK Ofens./Defens./Control**: Posición en cada ranking (1 = mejor)
            - **Score**: Puntuación normalizada de 1-10 en cada categoría  
            - **Promedio RK**: Media de las posiciones (menor = mejor jugador integral)
            
            **⚠️ Nota**: Esta comparación se realiza únicamente entre los jugadores que aparecen en esta tabla.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #7b1fa2;">', unsafe_allow_html=True)
            st.markdown("### 🎨 <span style='color: #7b1fa2'>Código de colores</span>", unsafe_allow_html=True)
            st.markdown("""
            - 🟢 **Verde**: 1º puesto o promedio ≤ 1.5 (excelente)
            - 🟡 **Amarillo**: Posiciones intermedias o promedio 1.5-2.5 (bueno)
            - 🔴 **Rojo**: Último puesto o promedio > 2.5 (mejorable)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div style="background-color: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #388e3c;">', unsafe_allow_html=True)
            st.markdown("### 💡 <span style='color: #388e3c'>Interpretación</span>", unsafe_allow_html=True)
            st.markdown("""
            - Un jugador con promedio 1.0 está 1º en todas las categorías
            - Un jugador con promedio 2.0 está en promedio 2º en todas las categorías
            - Útil para identificar al **jugador más completo** del grupo
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    # ==================== CONTEXTO COMPARATIVO ====================
    st.markdown('<div class="section-header">🎯 Contexto Comparativo</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtro por posición general
        pos_generales = df["position_gen_ESP1"].dropna().unique() if "position_gen_ESP1" in df.columns else []
        if len(pos_generales) > 0:
            pos_gen_sel = st.multiselect(
                "Posición general:",
                [""] + sorted(pos_generales)
            )
        else:
            pos_gen_sel = []
    
    with col2:
        # Filtro por minutos jugados
        min_min = int(df["minutes_on_field"].min())
        max_min = int(df["minutes_on_field"].max())
        min_jugados = st.slider(
            "Minutos mínimos jugados:",
            min_value=min_min,
            max_value=max_min,
            value=min_min
        )
    
    with col3:
        # Filtro por edad
        edad_min = int(df["age"].min())
        edad_max = int(df["age"].max())
        rango_edad = st.slider(
            "Rango de edad:",
            min_value=edad_min,
            max_value=edad_max,
            value=(edad_min, edad_max)
        )
    
    # Crear muestra comparativa
    muestra = df.copy()
    if ligas_sel:
        muestra = muestra[muestra["League"].isin(ligas_sel)]
    if pos_gen_sel:
        muestra = muestra[muestra["position_gen_ESP1"].isin(pos_gen_sel)]
    muestra = muestra[muestra["minutes_on_field"] >= min_jugados]
    muestra = muestra[(muestra["age"] >= rango_edad[0]) & (muestra["age"] <= rango_edad[1])]


    
    # Mostrar información de la muestra
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jugadores en muestra", len(muestra))
    with col2:
        if not muestra.empty:
            st.metric("Edad promedio", f"{muestra['age'].mean():.1f}")
    with col3:
        if not muestra.empty:
            st.metric("Minutos promedio", f"{muestra['minutes_on_field'].mean():.0f}")
    
    
    # ==================== SCATTERPLOTS POR CATEGORÍA (2 FILAS x 3 COLUMNAS) ====================
    st.markdown('<div class="section-header">📈 Análisis de Dispersión por Categoría</div>', unsafe_allow_html=True)

    # Crear 2 filas con 3 columnas cada una
    st.markdown("#### Configuración de Métricas por Categoría")

    # Primera fila - Selección de métricas
    col1, col2, col3 = st.columns(3)

    scatterplot_configs = {}

    with col1:
        st.markdown('<div class="categoria-header ofensiva">🎯 OFENSIVA</div>', unsafe_allow_html=True)
        metricas_ofensiva = [m for m in METRICAS_CATEGORIAS['Ofensiva'] if m in df_sel.columns]
        if len(metricas_ofensiva) >= 2:
            x1_of = st.selectbox("Eje X:", metricas_ofensiva, key="x1_ofensiva")
            y1_of = st.selectbox("Eje Y:", metricas_ofensiva, key="y1_ofensiva", index=min(1, len(metricas_ofensiva)-1))
            scatterplot_configs['Ofensiva'] = [(x1_of, y1_of)]

    with col2:
        st.markdown('<div class="categoria-header defensiva">🛡️ DEFENSIVA</div>', unsafe_allow_html=True)
        metricas_defensiva = [m for m in METRICAS_CATEGORIAS['Defensiva'] if m in df_sel.columns]
        if len(metricas_defensiva) >= 2:
            x1_def = st.selectbox("Eje X:", metricas_defensiva, key="x1_defensiva")
            y1_def = st.selectbox("Eje Y:", metricas_defensiva, key="y1_defensiva", index=min(1, len(metricas_defensiva)-1))
            scatterplot_configs['Defensiva'] = [(x1_def, y1_def)]

    with col3:
        st.markdown('<div class="categoria-header control">⚽ CONTROL</div>', unsafe_allow_html=True)
        metricas_control = [m for m in METRICAS_CATEGORIAS['Control'] if m in df_sel.columns]
        if len(metricas_control) >= 2:
            x1_con = st.selectbox("Eje X:", metricas_control, key="x1_control")
            y1_con = st.selectbox("Eje Y:", metricas_control, key="y1_control", index=min(1, len(metricas_control)-1))
            scatterplot_configs['Control'] = [(x1_con, y1_con)]

    # Segunda fila - Selección adicional de métricas
    col1, col2, col3 = st.columns(3)

    with col1:
        if len(metricas_ofensiva) >= 2:
            x2_of = st.selectbox("Eje X (2º):", metricas_ofensiva, key="x2_ofensiva", index=min(2, len(metricas_ofensiva)-1))
            y2_of = st.selectbox("Eje Y (2º):", metricas_ofensiva, key="y2_ofensiva", index=min(3, len(metricas_ofensiva)-1))
            scatterplot_configs['Ofensiva'].append((x2_of, y2_of))

    with col2:
        if len(metricas_defensiva) >= 2:
            x2_def = st.selectbox("Eje X (2º):", metricas_defensiva, key="x2_defensiva", index=min(2, len(metricas_defensiva)-1))
            y2_def = st.selectbox("Eje Y (2º):", metricas_defensiva, key="y2_defensiva", index=min(3, len(metricas_defensiva)-1))
            scatterplot_configs['Defensiva'].append((x2_def, y2_def))

    with col3:
        if len(metricas_control) >= 2:
            x2_con = st.selectbox("Eje X (2º):", metricas_control, key="x2_control", index=min(2, len(metricas_control)-1))
            y2_con = st.selectbox("Eje Y (2º):", metricas_control, key="y2_control", index=min(3, len(metricas_control)-1))
            scatterplot_configs['Control'].append((x2_con, y2_con))

    # Crear los scatterplots en grid 2x3
    st.markdown("#### Gráficos de Dispersión Interactivos")
    st.info("💡 **Pasa el cursor sobre cualquier punto** para ver información del jugador (nombre, equipo, métricas, posición, edad)")

    # Primera fila de gráficos
    col1, col2, col3 = st.columns(3)

    for i, (categoria, color) in enumerate(zip(['Ofensiva', 'Defensiva', 'Control'], 
                                            [COLORES_CATEGORIAS['Ofensiva'], COLORES_CATEGORIAS['Defensiva'], COLORES_CATEGORIAS['Control']])):
        if categoria in scatterplot_configs and scatterplot_configs[categoria]:
            x_metric, y_metric = scatterplot_configs[categoria][0]
            
            # Usar la versión interactiva con Plotly
            fig = crear_scatterplot_interactivo(muestra, df_sel, x_metric, y_metric, categoria, color)
            
            if fig:
                if i == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                elif i == 1:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col3:
                        st.plotly_chart(fig, use_container_width=True)

    # Segunda fila de gráficos
    col1, col2, col3 = st.columns(3)

    for i, (categoria, color) in enumerate(zip(['Ofensiva', 'Defensiva', 'Control'], 
                                            [COLORES_CATEGORIAS['Ofensiva'], COLORES_CATEGORIAS['Defensiva'], COLORES_CATEGORIAS['Control']])):
        if categoria in scatterplot_configs and len(scatterplot_configs[categoria]) > 1:
            x_metric, y_metric = scatterplot_configs[categoria][1]
            
            # Usar la versión interactiva con Plotly
            fig = crear_scatterplot_interactivo(muestra, df_sel, x_metric, y_metric, categoria, color)
            
            if fig:
                if i == 0:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                elif i == 1:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col3:
                        st.plotly_chart(fig, use_container_width=True)

    # Información sobre los scatterplots
    with st.expander("📈 Información sobre los Scatterplots Interactivos"):
        st.markdown("""
        **🖱️ Características de los Gráficos Interactivos:**
        
        **Funcionalidades disponibles:**
        - **Hover dinámico**: Pasa el cursor sobre CUALQUIER punto para ver información completa
        - **Zoom**: Usa la rueda del ratón o selecciona un área para hacer zoom
        - **Pan**: Arrastra para moverte por el gráfico
        - **Reset**: Doble clic para volver a la vista original
        - **Descarga**: Botón de cámara para guardar el gráfico como imagen
        
        **Información mostrada en hover:**
        - 📝 **Nombre del jugador**
        - ⚽ **Equipo**
        - 📊 **Valores exactos de las métricas X e Y**
        - 📍 **Posición** (si está disponible)
        - 🎂 **Edad** (si está disponible)
        - ⏱️ **Minutos jugados** (si está disponible)
        
        **Código de colores:**
        - **Puntos grises**: Otros jugadores de la muestra comparativa
        - **Puntos de colores con nombres**: Jugadores seleccionados para análisis
        - **⭐ en hover**: Indica jugador seleccionado
        
        **📏 Líneas de referencia:**
        - **Línea Roja Horizontal**: Promedio de la métrica del eje Y
        - **Línea Azul Vertical**: Promedio de la métrica del eje X
        
        **🎯 Interpretación por Cuadrantes:**
        - **Superior Derecha**: Por encima del promedio en ambas métricas (🟢 Excelente)
        - **Superior Izquierda**: Alto en Y, bajo en X (🟡 Especialista Y)
        - **Inferior Derecha**: Alto en X, bajo en Y (🟡 Especialista X)  
        - **Inferior Izquierda**: Por debajo del promedio en ambas (🔴 Mejorable)
        
        **💡 Tip**: Usa el hover para descubrir jugadores interesantes en la muestra que podrías querer analizar más a fondo.
        """)
    
    # ==================== ANÁLISIS PIZZA CHARTS TRIPLE ====================
    st.markdown('<div class="section-header">🍕 Análisis Integral por Jugador</div>', unsafe_allow_html=True)

# Selector de jugador
jugador_analisis = st.selectbox(
    "Selecciona un jugador para análisis integral:",
    seleccionados
)

if jugador_analisis:
    jugador_data = df_sel[df_sel["jugador_id"] == jugador_analisis].iloc[0]
    
    # INFORMACIÓN COMÚN DEL JUGADOR
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Jugador", jugador_analisis.split("(")[0].strip())
    with col2:
        st.metric("⚽ Equipo", jugador_analisis.split("(")[1].replace(")", "").strip())
    with col3:
        if "age" in jugador_data:
            st.metric("🎂 Edad", f"{jugador_data.get('age', 'N/A')}")
    with col4:
        if "primary_position_ESP" in jugador_data:
            st.metric("📍 Posición", jugador_data.get('primary_position_ESP', 'N/A'))
    
    # ==================== KPIs DEL JUGADOR ====================
    st.markdown("### 📊 Indicadores de Rendimiento (Percentiles)")
    
    # Calcular KPIs usando la muestra de referencia o todo el dataset
    df_referencia_kpis = muestra if not muestra.empty else df
    ataque_kpi, defensa_kpi, control_kpi, total_kpi = calcular_kpis_para_un_jugador(
        jugador_data, df_referencia_kpis, METRICAS_CATEGORIAS
    )
    
    # Mostrar KPIs en tarjetas visuales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r, g, b = color_for_kpi(ataque_kpi)
        st.markdown(f"""
        <div style="background-color: rgb({r},{g},{b}); padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #333;">
            <h3 style="margin: 0; font-size: 14px;">🎯 ATAQUE</h3>
            <h2 style="margin: 5px 0; font-size: 24px; font-weight: bold;">{ataque_kpi:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        r, g, b = color_for_kpi(defensa_kpi)
        st.markdown(f"""
        <div style="background-color: rgb({r},{g},{b}); padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #333;">
            <h3 style="margin: 0; font-size: 14px;">🛡️ DEFENSA</h3>
            <h2 style="margin: 5px 0; font-size: 24px; font-weight: bold;">{defensa_kpi:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        r, g, b = color_for_kpi(control_kpi)
        st.markdown(f"""
        <div style="background-color: rgb({r},{g},{b}); padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #333;">
            <h3 style="margin: 0; font-size: 14px;">⚽ CONTROL</h3>
            <h2 style="margin: 5px 0; font-size: 24px; font-weight: bold;">{control_kpi:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        r, g, b = color_for_kpi(total_kpi)
        st.markdown(f"""
        <div style="background-color: rgb({r},{g},{b}); padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #333;">
            <h3 style="margin: 0; font-size: 14px;">🏆 TOTAL</h3>
            <h2 style="margin: 5px 0; font-size: 24px; font-weight: bold;">{total_kpi:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Interpretación de los KPIs
    st.markdown("#### 💡 Interpretación de KPIs:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Rangos de percentiles:**
        - 🟢 **75-100**: Excelente (Top 25%)
        - 🟡 **60-74**: Muy bueno (Top 40%)  
        - 🟠 **45-59**: Bueno (Promedio)
        - 🔴 **30-44**: Por debajo del promedio
        - ⚫ **0-29**: Necesita mejorar
        """)
    
    with col2:
        # Mostrar la posición general del jugador
        mejor_categoria = max([("Ataque", ataque_kpi), ("Defensa", defensa_kpi), ("Control", control_kpi)], key=lambda x: x[1])
        peor_categoria = min([("Ataque", ataque_kpi), ("Defensa", defensa_kpi), ("Control", control_kpi)], key=lambda x: x[1])
        
        st.markdown(f"""
        **🎯 Análisis del jugador:**
        - **Fortaleza**: {mejor_categoria[0]} ({mejor_categoria[1]:.1f})
        - **Área de mejora**: {peor_categoria[0]} ({peor_categoria[1]:.1f})
        - **Rendimiento global**: {total_kpi:.1f} percentil
        """)
    
    st.markdown("---")
    
    # ==================== 3 PIZZA CHARTS EN FILA ====================
    st.markdown("### 🎯 Perfil Integral: Ofensiva | Defensiva | Control")
    
    col1, col2, col3 = st.columns(3)
    
    df_referencia_global = muestra if not muestra.empty else df
    
    # Guardar Z-scores por categoría para el resumen
    zscores_por_categoria = {}
    
    for i, (categoria, metricas) in enumerate(METRICAS_CATEGORIAS.items()):
        # Filtrar métricas disponibles
        metricas_disponibles = [m for m in metricas if m in df_sel.columns]
        
        if not metricas_disponibles:
            continue
        
        # CÁLCULO 1: Percentiles globales
        percentiles_globales = []
        for metrica in metricas_disponibles:
            if metrica in jugador_data and metrica in df_referencia_global.columns:
                valor = jugador_data[metrica]
                if pd.notna(valor):
                    percentil = percentileofscore(df_referencia_global[metrica].dropna(), valor)
                    percentiles_globales.append(min(100, max(0, int(percentil))))
                else:
                    percentiles_globales.append(50)
            else:
                percentiles_globales.append(50)
        
        # CÁLCULO 2: Z-score vs grupo
        percentiles_zscore, zscores_raw = calcular_zscore_jugador_vs_grupo(
            df_sel, jugador_data, metricas_disponibles
        )
        
        # Guardar para resumen
        zscores_por_categoria[categoria] = zscores_raw
        
        # Crear pizza chart
        metricas_limpias = [m.replace("_", " ").title() for m in metricas_disponibles]
        fig_pizza = crear_pizza_chart_categoria(
            metricas=metricas_limpias,
            valores_jugador=percentiles_globales,
            valores_promedio=percentiles_zscore,
            jugador=jugador_analisis.split("(")[0].strip(),
            categoria=categoria,
            color_principal=COLORES_CATEGORIAS[categoria]
        )
        
        # Mostrar en la columna correspondiente CON LEYENDA
        if i == 0:  # Ofensiva
            with col1:
                if fig_pizza:
                    st.pyplot(fig_pizza)
                st.metric(f"🎯 Promedio Z-score", f"{np.mean(zscores_raw):.2f}")
                
                # LEYENDA ESPECÍFICA
                leyenda_info = crear_leyenda_radar_categoria('Ofensiva')
                with st.expander(f"📖 {leyenda_info['titulo']}"):
                    st.markdown(leyenda_info['explicacion'])
                    
        elif i == 1:  # Defensiva
            with col2:
                if fig_pizza:
                    st.pyplot(fig_pizza)
                st.metric(f"🛡️ Promedio Z-score", f"{np.mean(zscores_raw):.2f}")
                
                # LEYENDA ESPECÍFICA
                leyenda_info = crear_leyenda_radar_categoria('Defensiva')
                with st.expander(f"📖 {leyenda_info['titulo']}"):
                    st.markdown(leyenda_info['explicacion'])
                    
        else:  # Control
            with col3:
                if fig_pizza:
                    st.pyplot(fig_pizza)
                st.metric(f"⚽ Promedio Z-score", f"{np.mean(zscores_raw):.2f}")
                
                # LEYENDA ESPECÍFICA
                leyenda_info = crear_leyenda_radar_categoria('Control')
                with st.expander(f"📖 {leyenda_info['titulo']}"):
                    st.markdown(leyenda_info['explicacion'])
    
    # ==================== RESUMEN INTEGRAL ÚNICO ====================
    st.markdown("### 📋 Resumen Integral")
    
    # Calcular Z-scores promedio por categoría
    col1, col2, col3 = st.columns(3)
    
    zscore_general_valores = []
    
    for i, (categoria, zscores_raw) in enumerate(zscores_por_categoria.items()):
        zscore_promedio = np.mean(zscores_raw)
        zscore_general_valores.append(zscore_promedio)
        
        if i == 0:  # Ofensiva
            with col1:
                st.metric(
                    f"🎯 {categoria}",
                    f"{zscore_promedio:.2f}",
                    delta=interpretar_zscore(zscore_promedio)
                )
        elif i == 1:  # Defensiva
            with col2:
                st.metric(
                    f"🛡️ {categoria}",
                    f"{zscore_promedio:.2f}",
                    delta=interpretar_zscore(zscore_promedio)
                )
        else:  # Control
            with col3:
                st.metric(
                    f"⚽ {categoria}",
                    f"{zscore_promedio:.2f}",
                    delta=interpretar_zscore(zscore_promedio)
                )
    
    # Interpretación automática integral
    if zscore_general_valores:
        zscore_general = np.mean(zscore_general_valores)
        
        if zscore_general > 1:
            st.success(f"🌟 **{jugador_analisis.split('(')[0].strip()}** muestra un rendimiento **integral muy superior** al grupo")
        elif zscore_general > 0.5:
            st.success(f"👍 **{jugador_analisis.split('(')[0].strip()}** muestra un rendimiento **integral superior** al grupo")
        elif zscore_general > -0.5:
            st.info(f"📊 **{jugador_analisis.split('(')[0].strip()}** muestra un rendimiento **integral promedio** respecto al grupo")
        else:
            st.warning(f"📉 **{jugador_analisis.split('(')[0].strip()}** muestra un rendimiento **integral inferior** al grupo")

# ==================== LEYENDA GENERAL ADICIONAL ====================
with st.expander("🎨 Guía de Interpretación de Colores por Categoría"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 OFENSIVA**
        - 🔴 **Rojo**: Percentil global
        - 🟢 **Verde**: Z-score grupal
        - Mide: Goles, asistencias, creatividad
        """)
    
    with col2:
        st.markdown("""
        **🛡️ DEFENSIVA**  
        - 🟢 **Verde Oscuro**: Percentil global
        - 🟠 **Naranja**: Z-score grupal
        - Mide: Duelos, intercepciones, solidez
        """)
    
    with col3:
        st.markdown("""
        **⚽ CONTROL**
        - 🔵 **Azul**: Percentil global  
        - 🟢 **Verde**: Z-score grupal
        - Mide: Pases, precisión, construcción
        """)
            

# ==================== EXPORTAR DATOS ====================
if seleccionados:
    st.markdown('<div class="section-header">💾 Exportar Análisis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Preparar datos para exportar (todas las métricas disponibles)
        todas_metricas = []
        for metricas in METRICAS_CATEGORIAS.values():
            todas_metricas.extend([m for m in metricas if m in df_sel.columns])
        
        df_export = df_sel[["jugador_id"] + todas_metricas].copy()
        csv = df_export.to_csv(index=False, encoding='utf-8')
        
        st.download_button(
            label="📁 Descargar análisis por categorías (CSV)",
            data=csv,
            file_name=f"analisis_categorias_{len(seleccionados)}_jugadores.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("📊 Generar reporte integral"):
            with st.spinner("Generando informe PDF..."):

                # === ⬇️ Generar y guardar pizza charts de cada jugador antes del PDF
                for jugador_id in seleccionados:
                    jugador = df_sel[df_sel["jugador_id"] == jugador_id].iloc[0]
                    
                    # CREAR NOMBRE CONSISTENTE CON EL PDF
                    nombre_jugador = jugador["name"]
                    equipo_jugador = jugador["last_club_name"]
                    safe_name = safe_filename(f"{nombre_jugador} {equipo_jugador}")
                    
                    for categoria, metricas in METRICAS_CATEGORIAS.items():
                        metricas_disponibles = [m for m in metricas if m in df_sel.columns]
                        if not metricas_disponibles:
                            continue

                        metricas_limpias = [m.replace("_", " ").title() for m in metricas_disponibles]
                        percentiles_globales = [
                            round(percentileofscore(df[metrica].dropna(), jugador[metrica]), 2) if pd.notna(jugador[metrica]) else 50.0
                            for metrica in metricas_disponibles
                        ]
                        percentiles_zscore, _ = calcular_zscore_jugador_vs_grupo(df_sel, jugador, metricas_disponibles)
                        percentiles_zscore = [round(val, 2) for val in percentiles_zscore]

                        fig_pizza = crear_pizza_chart_categoria(
                            metricas=metricas_limpias,
                            valores_jugador=percentiles_globales,
                            valores_promedio=percentiles_zscore,
                            jugador=nombre_jugador,
                            categoria=categoria,
                            color_principal=COLORES_CATEGORIAS[categoria]
                        )

                        if fig_pizza:
                            pizza_path = os.path.join(PIZZA_DIR, f"{safe_name}_{categoria}.png")
                            fig_pizza.savefig(pizza_path, bbox_inches='tight')
                            plt.close(fig_pizza)

                pdf = PDF()

                # Crear carpetas si no existen
                os.makedirs("figures/tables", exist_ok=True)
                os.makedirs("figures/scatterplots", exist_ok=True)

                # === 1. Portada
                pdf.add_portada("Informe Comparativa Laterales", "figures/logos/IMG_2027-removebg-preview.png")

                # === 2. Índice (con jugadores ordenados por KPI)
                # Primero calculamos los KPIs para crear el índice ordenado
                jugadores_para_indice = []
                df_referencia_kpis = muestra if not muestra.empty else df

                for jugador_id in seleccionados:
                    jugador_row = df_sel[df_sel["jugador_id"] == jugador_id].iloc[0]
                    nombre_jugador = jugador_row["name"]
                    
                    try:
                        _, _, _, total_kpi = calcular_kpis_para_un_jugador(
                            jugador_row, df_referencia_kpis, METRICAS_CATEGORIAS
                        )
                        jugadores_para_indice.append((nombre_jugador, total_kpi))
                    except Exception as e:
                        jugadores_para_indice.append((nombre_jugador, 0))

                # Ordenar por KPI total descendente para el índice
                jugadores_para_indice.sort(key=lambda x: x[1], reverse=True)
                nombres_ordenados = [nombre for nombre, kpi in jugadores_para_indice]

                pdf.add_index_page(nombres_ordenados)

                # === 3. Rankings por categoría (SOLO las 3 categorías)
                ranking_categorias_figs = []

                # PRIMERO: Crear las 3 tablas de categorías
                for categoria, metricas in METRICAS_CATEGORIAS.items():
                    metricas_disponibles = [m for m in metricas if m in df_sel.columns]
                    if not metricas_disponibles:
                        continue
                    df_categoria = df_sel[["jugador_id"] + metricas_disponibles].copy()
                    df_norm_cat = normalizar_datos(df_categoria, metricas_disponibles)
                    df_norm_cat = calcular_scoring(df_norm_cat, metricas_disponibles)
                    fig_tabla = crear_tabla_comparativa_categoria(df_norm_cat, metricas_disponibles, categoria, COLORES_CATEGORIAS[categoria])
                    if fig_tabla:
                        path = f"figures/tables/ranking_{categoria.lower()}.png"
                        fig_tabla.savefig(path, bbox_inches="tight")
                        ranking_categorias_figs.append(path)
                        plt.close(fig_tabla)

                # SEGUNDO: Combinar SOLO las 3 tablas de categorías en una hoja
                if len(ranking_categorias_figs) == 3:
                    from PIL import Image
                    imgs = [Image.open(f) for f in ranking_categorias_figs]
                    widths, heights = zip(*(i.size for i in imgs))
                    max_width = max(widths)
                    total_height = sum(heights)
                    
                    # Crear imagen combinada con las 3 categorías en vertical
                    combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                    y_offset = 0
                    for img in imgs:
                        # Centrar horizontalmente si la imagen es más pequeña
                        x_offset = (max_width - img.size[0]) // 2
                        combined.paste(img, (x_offset, y_offset))
                        y_offset += img.size[1]
                    
                    combined.save("figures/tables/rankings_categorias.png")
                    
                    # Añadir al PDF la página de rankings por categoría
                    pdf.add_image_page("Rankings por Categoria", "figures/tables/rankings_categorias.png")

                # === 4. Ranking Consolidado en hoja separada CON LEYENDA
                fig_consolidada = crear_tabla_rankings_consolidada(df_sel, METRICAS_CATEGORIAS, COLORES_CATEGORIAS)
                if fig_consolidada:
                    path_consolidada = "figures/tables/ranking_consolidado.png"
                    fig_consolidada.savefig(path_consolidada, bbox_inches="tight", dpi=150)
                    plt.close(fig_consolidada)
                    
                    # Crear imagen con tabla + leyenda usando PIL
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # Cargar la tabla
                    tabla_img = Image.open(path_consolidada)
                    tabla_width, tabla_height = tabla_img.size
                    
                    # Crear imagen con tabla + leyenda EN COLUMNAS usando PIL
                    from PIL import Image, ImageDraw, ImageFont

                    # Cargar la tabla
                    tabla_img = Image.open(path_consolidada)
                    tabla_width, tabla_height = tabla_img.size

                    # Crear nueva imagen con tabla + espacio para leyenda EN COLUMNAS
                    leyenda_height = 120  # Menos altura porque será en columnas
                    nueva_altura = tabla_height + leyenda_height
                    nueva_width = max(tabla_width, 800)

                    combined_img = Image.new('RGB', (nueva_width, nueva_altura), (255, 255, 255))

                    # Pegar tabla centrada arriba
                    x_offset_tabla = (nueva_width - tabla_width) // 2
                    combined_img.paste(tabla_img, (x_offset_tabla, 0))

                    # Añadir texto de leyenda EN 3 COLUMNAS
                    draw = ImageDraw.Draw(combined_img)

                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                        font_bold = ImageFont.truetype("arialbd.ttf", 22)
                    except:
                        font = ImageFont.load_default()
                        font_bold = ImageFont.load_default()

                    y_pos = tabla_height + 20
                    col_width = nueva_width // 3

                    # COLUMNA 1: COLUMNAS DE LA TABLA
                    x1 = 20
                    draw.text((x1, y_pos), "COLUMNAS DE LA TABLA:", fill=(25, 118, 210), font=font_bold)
                    y_temp = y_pos + 25
                    textos_col1 = [
                        "• RK Ofens./Defens./Control: Posicion en cada ranking (1 = mejor)",
                        "• Score: Puntuacion normalizada de 1-10 en cada categoria",
                        "• Promedio RK: Media de las posiciones (menor = mejor jugador integral)",
                        "",
                    ]
                    for texto in textos_col1:
                        draw.text((x1, y_temp), texto, fill=(0, 0, 0), font=font)
                        y_temp += 22

                    # COLUMNA 2: CODIGO DE COLORES  
                    x2 = col_width + 20
                    draw.text((x2, y_pos), "CODIGO DE COLORES:", fill=(123, 31, 162), font=font_bold)
                    y_temp = y_pos + 25
                    textos_col2 = [
                        "• Verde: 1er puesto o promedio <= 1.5 (excelente)",
                        "• Amarillo: Posiciones intermedias o promedio 1.5-2.5 (bueno)",
                        "• Rojo: Ultimo puesto o promedio > 2.5 (mejorable)",
                        "",
                    ]
                    for texto in textos_col2:
                        draw.text((x2, y_temp), texto, fill=(0, 0, 0), font=font)
                        y_temp += 22

                    # COLUMNA 3: INTERPRETACION
                    x3 = 2 * col_width + 20  
                    draw.text((x3, y_pos), "INTERPRETACION:", fill=(56, 142, 60), font=font_bold)
                    y_temp = y_pos + 25
                    textos_col3 = [
                        "• Un jugador con promedio 1.0 esta 1o en todas las categorias",
                        "• Un jugador con promedio 2.0 esta en promedio 2o en todas las categorias",
                        "• Util para identificar al jugador mas completo del grupo"
                    ]
                    for texto in textos_col3:
                        draw.text((x3, y_temp), texto, fill=(0, 0, 0), font=font)
                        y_temp += 22

                    # Guardar imagen combinada
                    path_consolidada_final = "figures/tables/ranking_consolidado_con_leyenda.png"
                    combined_img.save(path_consolidada_final)
                    
                    # Añadir al PDF
                    pdf.add_image_page("Ranking Consolidado", path_consolidada_final)

                # === 5-7. Scatterplots
                for categoria, pares in scatterplot_configs.items():
                    for i, (x, y) in enumerate(pares):
                        fig = crear_scatterplot_interactivo(muestra, df_sel, x, y, categoria, COLORES_CATEGORIAS[categoria])
                        if fig:
                            ruta = f"figures/scatterplots/scatter_{categoria.lower()}_{i+1}.png"
                            fig.write_image(ruta, scale=2)  # ✅ Guardar tal cual está
                                                    
                            # GUARDAR CON TÍTULO TEMPORAL SIMPLE
                            fig_temp = fig
                            fig_temp.update_layout(title=f"{categoria} - Graf {i+1}")  # Título temporal corto
                            fig_temp.write_image(ruta, scale=2)
                            plt.close('all')
                # Cargar scatterplots desde disco y añadir al PDF
                scatters = {
                    "Ofensivo": {
                        "imgs": ["scatter_ofensiva_1.png", "scatter_ofensiva_2.png"],
                        "captions": [
                            "Non Penalty Goal Avg vs Head Goals Avg\n6 seleccionados | 111 jugadores",
                            "Xg Shot Avg vs Shots On Target Percent\n6 seleccionados | 111 jugadores"
                            
                        ]
                    },
                    "Defensivo": {
                        "imgs": ["scatter_defensiva_1.png", "scatter_defensiva_2.png"],
                        "captions": [
                            "Defensive Duels avg vs Defensive Duels Won\n6 seleccionados | 111 jugadores",
                            "Aerial Duels avg vs Aerial Duels Won\n6 seleccionados | 111 jugadores"
                        ]
                    },
                    "Control": {
                        "imgs": ["scatter_control_1.png", "scatter_control_2.png"],
                        "captions": [
                            " Passes Avg vs Accurate Passes Percent\n6 seleccionados | 111 jugadores",
                            "Long Passes Avg vs Successful Long Passes Percent\n6 seleccionados | 111 jugadores"
                        ]
                    }
                }

                for categoria, data in scatters.items():
                    path1 = os.path.join("figures", "scatterplots", data["imgs"][0])
                    path2 = os.path.join("figures", "scatterplots", data["imgs"][1])
                    cap1 = data["captions"][0]
                    cap2 = data["captions"][1]
                    
                    if os.path.exists(path1) and os.path.exists(path2):
                        pdf.add_scatterplots_page(f"Scatterplots: {categoria}", path1, path2, caption1=cap1, caption2=cap2)

                # === 8. Página de introducción para jugadores (DESPUÉS DE SCATTERPLOTS)
                pdf.add_player_intro_page()

                # === 9+. Páginas individuales por jugador (ORDENADAS POR KPI TOTAL)
                # Definir dataframe de referencia para KPIs (igual que en la app)
                df_referencia_kpis = muestra if not muestra.empty else df

                # Calcular KPIs para todos los jugadores y ordenar por KPI total
                jugadores_con_kpis = []

                for jugador_id in seleccionados:
                    jugador_row = df_sel[df_sel["jugador_id"] == jugador_id].iloc[0]
                    jugador_dict = jugador_row.to_dict()
                    
                    # Calcular KPIs para ordenamiento
                    try:
                        ataque_kpi, defensa_kpi, control_kpi, total_kpi = calcular_kpis_para_un_jugador(
                            jugador_row, df_referencia_kpis, METRICAS_CATEGORIAS
                        )
                        jugador_dict['kpi_total'] = total_kpi
                        jugadores_con_kpis.append(jugador_dict)
                    except Exception as e:
                        print(f"Error calculando KPIs para {jugador_id}: {e}")
                        jugador_dict['kpi_total'] = 0
                        jugadores_con_kpis.append(jugador_dict)

                # Ordenar por KPI total descendente (mayor a menor)
                jugadores_con_kpis.sort(key=lambda x: x['kpi_total'], reverse=True)

                # Generar páginas en orden de KPI
                for jugador_dict in jugadores_con_kpis:
                    pdf.add_player_page(jugador_dict, df_referencia_kpis, METRICAS_CATEGORIAS)
                
                # === GENERAR PÁGINA FINAL CON FONDO COMPLETO SIN FOOTER ===
                IMAGES_DIR = os.path.join("figures", "images")
                os.makedirs(IMAGES_DIR, exist_ok=True)
                estadio_path = "/Users/macmontxinho/Desktop/Teams/Wyscout/figures/logos/Estadio_Unionistas.jpg"

                if os.path.exists(estadio_path):
                    pdf.skip_footer = True  # 🔴 Desactivar footer solo para esta
                    pdf.add_full_page_image(estadio_path)
                    pdf.skip_footer = False  # 🟢 Reactivar para futuras páginas
            
                # === GENERAR PÁGINA FINAL SIN FOOTER ===
                if os.path.exists(estadio_path):
                    pdf.skip_footer = True         # 🔴 Desactiva footer SOLO para esta página
                    pdf.add_full_page_image(estadio_path)
                    pdf.skip_footer = False        # 🟢 Reactiva footer para siguientes páginas (si hubiera más)
                                    
                # === 10. Guardar y descargar PDF
                output_path = os.path.join(PDF_OUTPUT_DIR, f"informe_integral_{len(seleccionados)}_jugadores.pdf")
                pdf.output(output_path)
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Descargar informe PDF",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime="application/pdf"
                    )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666; font-size: 0.9em;'>
        ⚽ Análisis por Categorías: Ofensiva | Defensiva | Control | Desarrollado con Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)

            
