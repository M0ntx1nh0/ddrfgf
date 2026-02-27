# reporte_pdf.py

import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import re
import unidecode
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

def safe_filename(name):
    name = unidecode.unidecode(name)
    name = re.sub(r"[^\w\s-]", "", name)
    name = name.replace(" ", "_")
    return name


PDF_OUTPUT_DIR = "pdf_exports"
FIG_TEMP_DIR = "figures"
PIZZA_DIR = os.path.join(FIG_TEMP_DIR, "pizzas")
SCATTER_DIR = os.path.join(FIG_TEMP_DIR, "scatterplots")
TABLE_DIR = os.path.join(FIG_TEMP_DIR, "tables")
IMAGES_DIR = os.path.join(tempfile.gettempdir(), "wyscout_pdf_images")

os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
os.makedirs(PIZZA_DIR, exist_ok=True)
os.makedirs(SCATTER_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Renombrar todos los archivos de radar en PIZZA_DIR para que sean compatibles
for filename in os.listdir(PIZZA_DIR):
    if filename.lower().endswith(".png"):
        try:
            name_part, category = filename.rsplit("_", 1)
            new_name = f"{safe_filename(name_part)}_{category}"
            old_path = os.path.join(PIZZA_DIR, filename)
            new_path = os.path.join(PIZZA_DIR, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"[Renombrado] {filename} → {new_name}")
        except Exception as e:
            print(f"[Error renombrando] {filename}: {e}")

def clean_text_latin1(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

from PIL import ImageEnhance

def generar_pagina_cierre_con_fondo(fondo_path, escudo_path, salida_path):
    """
    Genera una imagen con el fondo del estadio y el escudo centrado encima (resaltado),
    luego la guarda en salida_path.
    """
    try:
        fondo = Image.open(fondo_path).convert("RGBA")
        escudo = Image.open(escudo_path).convert("RGBA")

        # Redimensionar el fondo a tamaño A4 horizontal (3508x2480 px a 300 DPI aprox)
        fondo = fondo.resize((3508, 2480))

        # Hacer el fondo ligeramente transparente
        fondo.putalpha(160)

        # Redimensionar escudo proporcional
        ancho_escudo = 800
        factor = ancho_escudo / escudo.width
        alto_escudo = int(escudo.height * factor)
        escudo = escudo.resize((ancho_escudo, alto_escudo), Image.LANCZOS)

        # Calcular posición para centrar
        x = (fondo.width - escudo.width) // 2
        y = (fondo.height - escudo.height) // 2

        # Pegar el escudo sobre el fondo
        fondo.paste(escudo, (x, y), escudo)

        # Convertir a RGB para guardar en JPG
        fondo_rgb = fondo.convert("RGB")
        fondo_rgb.save(salida_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error generando imagen final: {e}")
        return False

# ==================== FUNCIONES KPI ====================

def calcular_kpis_para_un_jugador_pdf(jugador_row, df_referencia, metricas_categorias):
    """
    Calcula los KPIs de Ataque, Defensa, Control y Total para un jugador específico
    basado en percentiles respecto a un dataframe de referencia.
    """
    from scipy.stats import percentileofscore
    import pandas as pd
    import numpy as np
    
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
            if metrica in df_referencia.columns:
                try:
                    # Obtener valor del jugador
                    valor_jugador = jugador_row.get(metrica, 0)
                    if pd.isna(valor_jugador):
                        valor_jugador = 0
                    
                    # Calcular percentil respecto al dataframe de referencia
                    valores_referencia = df_referencia[metrica].fillna(0)
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

class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=10)
        self.skip_footer = False  # << NUEVA bandera

    def header(self):
        # Título centrado
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Busqueda de Laterales mediante Clustering', ln=False, align='C')
        
        # Texto a la derecha
        self.set_xy(200, 10)  # Posición derecha
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, 'ST Unionistas de Salamanca CF', ln=True, align='R')
        self.ln(5)

    def footer(self):
        # No mostrar footer si está desactivado
        if self.page_no() == 1 or self.skip_footer:
            return
            
        self.set_y(-25)
        
        # ===== LADO IZQUIERDO: Autor y fuente en una misma línea =====
        self.set_font('Arial', '', 8)
        self.set_xy(15, self.get_y())
        self.cell(35, 10, 'Informe realizado por: ', ln=False)

        # Nombre en negrita
        self.set_font('Arial', 'B', 8)
        self.cell(30, 10, 'Ramón Codesido', ln=False)

        # Separador
        self.set_font('Arial', '', 8)
        self.cell(10, 10, '| Data: ', ln=False)

        # Fuente en negrita
        self.set_font('Arial', 'B', 8)
        self.cell(30, 10, 'Wyscout T.24/25', ln=False)
        
        # ===== CENTRO: Número de página =====
        self.set_font('Arial', 'I', 8)
        page_text = f'Pagina {self.page_no()}'
        page_width = self.get_string_width(page_text)
        center_x = (297 - page_width) / 2  # 297mm = ancho A4 horizontal
        self.set_xy(center_x, self.get_y())
        self.cell(page_width, 10, page_text, ln=False, align='C')
        
        # ===== DERECHA: Logo =====
        try:
            logo_path = "/Users/macmontxinho/Desktop/Teams/Wyscout/figures/logos/Escudo_Of.png"
            if os.path.exists(logo_path):
                # Probar diferentes tamaños y posiciones
                self.image(logo_path, x=250, y=self.get_y() - 3, w=22)
                print(f"[Success] Logo cargado: {logo_path}")
            else:
                print(f"[Error] Logo no encontrado en: {logo_path}")
                # Texto alternativo si no hay logo
                self.set_xy(250, -20)
                self.set_font('Arial', '', 8)
                self.cell(25, 10, '[LOGO]', ln=False, align='C')
        except Exception as e:
            print(f"[Error] Excepción cargando logo: {e}")

    def add_portada(self, titulo, logo_path=None):
        self.add_page()

        # Fondo negro
        self.set_fill_color(0, 0, 0)
        self.rect(0, 0, 297, 210, 'F')  # Página A4 horizontal

        # Logo centrado arriba
        if logo_path and os.path.exists(logo_path):
            self.image(logo_path, x=100, y=50, w=90)  # Ajusta 'y' si quieres subir o bajar el logo

        # Texto blanco (título principal)
        self.set_text_color(255, 255, 255)
        self.set_font("Arial", "B", 30)
        self.set_y(140)
        self.cell(0, 15, titulo, align="C")

        # Subtítulo en blanco
        self.set_font("Arial", "", 20)
        self.set_y(self.get_y() + 14)
        self.cell(0, 10, "ST Unionistas FC - Temporada 25/26", align="C")

        # Restaurar color de texto para el resto del documento
        self.set_text_color(0, 0, 0)

    def add_index_page(self, jugadores_ordenados_por_kpi):
        self.add_page()
        self.set_font("Arial", "B", 18)
        self.cell(0, 10, "Indice", ln=True, align="C")
        self.ln(10)
        self.set_font("Arial", '', 12)
        
        # Páginas de análisis grupal
        self.cell(0, 8, "1. Rankings por Categoria", ln=True)
        self.cell(0, 8, "2. Ranking Consolidado", ln=True)
        self.cell(0, 8, "3. Scatterplots: Ofensiva", ln=True)
        self.cell(0, 8, "4. Scatterplots: Defensiva", ln=True)
        self.cell(0, 8, "5. Scatterplots: Control", ln=True)
        self.cell(0, 8, "6. Guia de Interpretacion", ln=True)
        self.ln(5)
        
        # Páginas individuales ordenadas por KPI
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, "PERFILES INDIVIDUALES (Ordenados por KPI Total):", ln=True)
        self.ln(2)
        
        self.set_font("Arial", '', 11)
        for i, jugador_nombre in enumerate(jugadores_ordenados_por_kpi, start=7):
            self.cell(0, 7, f"{i}. Perfil: {jugador_nombre}", ln=True)

    def add_image_page(self, title, image_path):
        self.add_page()
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, ln=True, align="C")
        self.ln(5)
        self.image(image_path, x=10, y=25, w=270)

    def add_player_intro_page(self):
        """Añade página de introducción explicando la estructura de análisis de jugadores"""
        self.add_page()
        
        # Título principal CENTRADO
        self.set_font("Arial", "B", 18)
        self.set_text_color(0, 100, 150)
        self.cell(0, 12, "Guia de Interpretacion: Perfiles Individuales de Jugadores", ln=True, align="C")
        
        # Subtítulo CENTRADO VERTICALMENTE
        self.set_y(45)
        self.set_font("Arial", "B", 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, "Estructura y Metodologia del Analisis Individual", ln=True, align="C")
        
        # ===== CONTENIDO CENTRADO EN BLOQUE =====
        # Calcular posiciones para centrar el conjunto
        ancho_total_contenido = 250  # Ancho total del contenido
        x_inicio = (297 - ancho_total_contenido) / 2  # Centrar en página A4 horizontal
        
        x_izq = x_inicio  # Columna izquierda
        x_der = x_inicio + 130  # Columna derecha (130mm de separación)
        y_contenido = 65  # Posición Y inicial del contenido
        
        # ===== COLUMNA IZQUIERDA (Secciones 1 y 2) =====
        
        # SECCIÓN 1 - IZQUIERDA
        self.set_xy(x_izq, y_contenido)
        self.set_font("Arial", "B", 10)
        self.set_text_color(50, 50, 150)
        self.cell(0, 6, "1. ESTRUCTURA DE LA PAGINA DE JUGADOR", ln=True)
        
        self.set_xy(x_izq, self.get_y() + 2)
        self.set_font("Arial", "", 9)
        self.set_text_color(0, 0, 0)
        contenido_estructura = [
            "- Informacion basica: Datos personales, equipo, posicion",
            "- Indicadores KPI: Cuatro metricas de rendimiento (0-100)",  
            "- Radar Charts: Tres graficos especializados del perfil"
        ]
        
        for linea in contenido_estructura:
            self.set_x(x_izq)
            self.cell(120, 4, linea, ln=True)  # Ancho fijo para columna
        
        # SECCIÓN 2 - IZQUIERDA  
        self.set_xy(x_izq, self.get_y() + 5)
        self.set_font("Arial", "B", 10)
        self.set_text_color(150, 50, 50)
        self.cell(0, 6, "2. INDICADORES DE RENDIMIENTO (KPIs)", ln=True)
        
        self.set_xy(x_izq, self.get_y() + 2)
        self.set_font("Arial", "", 9)
        self.set_text_color(0, 0, 0)
        contenido_kpis = [
            "Los KPIs muestran el rendimiento promedio del jugador",
            "vs. jugadores similares, excluyendose a si mismo:",
            "",
            "- Ataque: Metricas ofensivas (goles, asistencias, etc.)",
            "- Defensa: Metricas defensivas (duelos, intercepciones, etc.)",
            "- Control: Metricas de posesion y pase",
            "- Total: Media de las tres categorias anteriores",
            "",
            "Calculo: Se toman varias metricas, se calcula el percentil", 
            "del jugador en cada una vs. la muestra, y se promedian.",
            "",
            "Interpretacion: Verde (75-100): Top 25% | Amarillo (60-74): Top 40%",
            "Naranja (45-59): Promedio | Rojo (30-44): Bajo | Gris (<30): Muy bajo"
        ]
        
        for linea in contenido_kpis:
            self.set_x(x_izq)
            self.cell(120, 3.5, linea, ln=True)
        
        # ===== COLUMNA DERECHA (Sección 3) - ALINEADA CON SECCIÓN 1 =====
        self.set_xy(x_der, y_contenido)
        self.set_font("Arial", "B", 10) 
        self.set_text_color(50, 150, 50)
        self.cell(0, 6, "3. RADAR CHARTS: DOBLE COMPARACION", ln=True)
        
        self.set_xy(x_der, self.get_y() + 2)
        self.set_font("Arial", "", 9)
        self.set_text_color(0, 0, 0)
        contenido_radars = [
            "Cada radar muestra DOS comparaciones simultaneas:",
            "",
            "A) PERCENTIL GLOBAL (Area exterior):",
            "Posicion vs. toda la muestra filtrada",
            "- Ofensiva (Rojo): Metricas ofensivas globales",
            "- Defensiva (Verde): Metricas defensivas globales", 
            "- Control (Azul): Metricas de control globales",
            "",
            "B) Z-SCORE GRUPAL (Area interior):",
            "Posicion vs. jugadores seleccionados unicamente.",
            "El Z-Score mide que tan lejos esta el jugador",
            "de la media del grupo seleccionado.",
            "- Verde en radars: Desv. estandar vs. grupo",
            "- Valores 0-100: 0=muy por debajo, 100=muy arriba",
            "- Z-Score positivo: Por encima del promedio",
            "- Z-Score negativo: Por debajo del promedio",
            "",
            "Interpretacion clave:",
            "- Area exterior > interior: Destaca globalmente",
            "- Area interior > exterior: Destaca en el grupo"
        ]
        
        for linea in contenido_radars:
            self.set_x(x_der)
            if linea.startswith("A)") or linea.startswith("B)"):
                self.set_font("Arial", "B", 9)
                self.cell(120, 3.5, linea, ln=True)
                self.set_font("Arial", "", 9)
            else:
                self.cell(120, 3.5, linea, ln=True)
        
        # ===== PIE DE PÁGINA MÁS ARRIBA =====
        self.set_y(170)  # Subir para no solaparse con número de página
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Esta metodologia permite identificar fortalezas y areas de mejora,", ln=True, align="C")
        self.cell(0, 5, "facilitando decisiones en fichajes, desarrollo y planificacion tactica.", ln=True, align="C")

    def add_scatterplots_page(self, title, img1, img2, caption1="", caption2=""):
        self.add_page()
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True, align="C")
        
        # Insertar las imágenes
        self.image(img1, x=15, y=45, w=125)
        self.image(img2, x=155, y=45, w=125)

        # Insertar captions debajo de cada gráfico
        self.set_font("Arial", "", 10)
        self.set_xy(15, 170)  # debajo de imagen izquierda
        self.multi_cell(125, 5, caption1, align="C")

        self.set_xy(155, 170)  # debajo de imagen derecha
        self.multi_cell(125, 5, caption2, align="C")

    def draw_metric_cards_grid(self, metrics, x_start=60, y_start=30):
        self.set_font('Arial', '', 6)
        card_width = 42
        card_height = 14
        padding_x = 1.5
        padding_y = 1
        cols = 3

        x = x_start
        y = y_start

        for i, (label, value) in enumerate(metrics.items()):
            if i > 0 and i % cols == 0:
                y += card_height + 2
                x = x_start

            self.set_xy(x, y)
            self.set_fill_color(240, 240, 240)
            self.cell(card_width, card_height, '', 0, 0, 'L', 1)

            self.set_xy(x + padding_x, y + padding_y)
            self.set_font('Arial', 'B', 6)
            self.cell(card_width - 2 * padding_x, 3.5, label, 0, 2, 'L')

            if label in ("Nombre", "Equipo 24/25"):
                self.set_font('Arial', 'B', 10)
            else:
                self.set_font('Arial', '', 8)

            self.set_text_color(117, 117, 117)
            self.multi_cell(card_width - 2 * padding_x, 4, clean_text_latin1(str(value)), border=0, align="C")
            self.set_text_color(0, 0, 0)

            x += card_width + 3

    def draw_kpi_cards(self, ataque, defensa, control, total, y_start=90):
        etiquetas = ["Ataque", "Defensa", "Control", "Total"]
        valores = [ataque, defensa, control, total]

        colores = {
            "high": (168, 230, 168),        # Verde claro
            "medium_high": (255, 252, 151), # Amarillo
            "medium": (255, 182, 142),      # Naranja claro
            "medium_low": (231, 154, 154),  # Naranja oscuro
            "low": (229, 115, 115),         # Rojo
        }

        def color(valor):
            if valor >= 75:
                return colores["high"]
            elif valor >= 60:
                return colores["medium_high"]
            elif valor >= 45:
                return colores["medium"]
            elif valor >= 30:
                return colores["medium_low"]
            else:
                return colores["low"]

        self.set_xy(10, y_start - 10)
        self.set_font('Arial', 'B', 10)
        self.cell(0, 5, "Indicadores de rendimiento (percentiles)", ln=True, align='C')

        card_width = 28
        card_height = 12
        spacing = 8
        total_width = 4 * card_width + 3 * spacing
        x = (297 - total_width) / 2  # Centrar en página horizontal A4
        y = y_start

        for etiqueta, valor in zip(etiquetas, valores):
            r, g, b = color(valor)

            self.set_font('Arial', 'B', 8)
            self.set_text_color(0, 0, 0)
            self.set_xy(x, y)
            self.cell(card_width, 4, etiqueta, ln=2, align='C')

            self.set_fill_color(r, g, b)
            self.set_font('Arial', '', 10)
            self.set_xy(x, y + 5)
            self.cell(card_width, card_height, f"{valor:.1f}", border=1, ln=0, align='C', fill=True)

            x += card_width + spacing

        self.set_text_color(0, 0, 0)

    def add_player_page(self, player, df_referencia=None, metricas_categorias=None):
        self.add_page()

        # Imagen del jugador
        img_url = player.get("image", "")
        img_path = ""
        if img_url:
            try:
                response = requests.get(img_url, timeout=10)
                img = Image.open(BytesIO(response.content))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert("RGB")
                img_path = os.path.join(IMAGES_DIR, f"{player['name'].replace('/', '_')}.jpg")
                img.save(img_path)
            except Exception as e:
                print(f"Error descargando imagen: {e}")

        if img_path and os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    aspect_ratio = img.height / img.width
                display_width = 45
                display_height = display_width * aspect_ratio
                self.image(img_path, x=10, y=30, w=display_width, h=display_height)
            except Exception as e:
                print(f"Error mostrando imagen en PDF: {e}")
            finally:
                # Imagen temporal: se elimina tras insertarse en el PDF
                try:
                    os.remove(img_path)
                except Exception:
                    pass

        # Tarjetas de información
        metrics = {
            "Nombre": player["name"],
            "Equipo 24/25": player["last_club_name"],
            "Edad": player["age"],
            "Posición Específica": player["primary_position_ESP"],
            "Posición General": player["position_gen_ESP1"],
            "Minutos | Partidos": f'{player["minutes_on_field"]} | {player.get("total_matches", "-")}',
            "Altura | Peso | Pie": f'{player["height"]} | {player.get("weight", "-")} | {player.get("foot", "-")}',
            "Goles | Asistencias": f'{player.get("goals", 0)} | {player.get("assists", 0)}',
            "Valor": player.get("market_value_fmt", f"€{player.get('market_value', 0):,.0f}".replace(",", "."))
        }

        self.draw_metric_cards_grid(metrics)

        # === CALCULAR Y MOSTRAR KPIs ===
        if df_referencia is not None and metricas_categorias is not None:
            try:
                ataque, defensa, control, total = calcular_kpis_para_un_jugador_pdf(
                    player, df_referencia, metricas_categorias
                )
                # Dibujar tarjetas KPI debajo de las tarjetas de información
                self.draw_kpi_cards(ataque, defensa, control, total, y_start=90)
                radar_y_pos = 120  # Mover los radars más abajo
            except Exception as e:
                print(f"Error calculando KPIs: {e}")
                radar_y_pos = 110  # Posición original si falla
        else:
            radar_y_pos = 110  # Posición original sin KPIs

        # === Radar charts debajo de las tarjetas KPI ===
        radar_categories = ["Ofensiva", "Defensiva", "Control"]
        chart_width = 60
        spacing = 15
        total_width = len(radar_categories) * chart_width + (len(radar_categories) - 1) * spacing
        x_start = (297 - total_width) / 2  # 297mm = ancho de página A4 horizontal

        club = player.get("last_club_name", "")
        safe_name = safe_filename(f"{player['name']} {club}")

        for i, categoria in enumerate(radar_categories):
            filename = f"{safe_name}_{categoria}.png"
            pizza_path = os.path.join(PIZZA_DIR, filename)
            if os.path.exists(pizza_path):
                x = x_start + i * (chart_width + spacing)

                # Título encima del radar
                self.set_xy(x, radar_y_pos - 5)
                self.set_font("Arial", "B", 10)
                self.cell(chart_width, 5, categoria.upper(), align="C")

                # Radar chart
                self.image(pizza_path, x=x, y=radar_y_pos, w=chart_width)
            else:
                print(f"[Radar] No encontrado: {pizza_path}")
    
    def add_full_page_image(self, image_path):
        self.skip_footer = True  # 🔴 Activar antes de add_page
        self.add_page()
        self.image(image_path, x=0, y=0, w=297, h=210)
        self.skip_footer = False  # 🟢 Reactivar footer para siguientes páginas
# Para exportar:
# pdf.output(os.path.join(PDF_OUTPUT_DIR, "informe_integral.pdf"))
