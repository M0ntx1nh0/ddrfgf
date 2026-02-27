# Wyscout Scouting App

Aplicación de scouting en **Streamlit** para análisis de jugadores (1RFEF / 2RFEF), orientada a procesos formativos y toma de decisiones con contexto visual.

## Objetivo
Ofrecer una herramienta práctica para:
- priorizar jugadores con rankings por métricas,
- evaluar perfiles con scoring por rol,
- comparar rendimiento con scatterplots, radares y swarmplots,
- añadir contexto de percentiles y diferencias frente a referencias.

## Funcionalidades principales
- Home con introducción, logos institucionales y navegación por pestañas.
- **Rankings** por métricas con filtros (liga, grupo, posición, minutos, edad, partidos).
- **Scoring** por perfil:
  - ofensivo,
  - defensivo,
  - portero/control (porteros usan score portero; resto usan control).
- Búsqueda contextual por equipo y jugador en tablas de scoring/ranking.
- **3 scatterplots editables** con métricas seleccionables.
- **3 radares (Ofensivo, Defensivo, Control)** + swarmplots asociados (Q1-Q4) y resaltado de jugadores.
- Carga de datos desde **Google Drive privado** con Service Account (sin subir CSV sensibles al repositorio).

## Arquitectura del proyecto
- `app.py`: portada y presentación.
- `pages/home.py`: rankings, scoring y gráficos.
- `common/data_loader.py`: carga de datos (Drive privado/URL/local fallback).
- `common/sidebar_branding.py`: branding y contacto en sidebar.
- `common/theme.py`: estilo visual global.
- `common/image_utils.py`: utilidades de imagen (transparencias).

## Requisitos
- Python 3.10+
- pip

## Instalación local
```bash
git clone https://github.com/M0ntx1nh0/ddrfgf.git
cd ddrfgf
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuración de entorno (.env)
Usa `.env.example` como base:

```bash
cp .env.example .env
```

Variables recomendadas:
```env
# Opción recomendada (Drive privado)
WYSCOUT_DRIVE_FILE_ID=TU_FILE_ID
GOOGLE_SERVICE_ACCOUNT_FILE=config/google-service-account.json

# Fallback local
WYSCOUT_CSV_LOCAL_PATH=data/Jugadoresv1.csv
```

Alternativa (si no usas archivo JSON local):
```env
GOOGLE_SERVICE_ACCOUNT_JSON={...json completo en una sola línea...}
```

## Google Drive privado (Service Account)
1. Crear proyecto en Google Cloud.
2. Activar **Google Drive API**.
3. Crear Service Account y descargar clave JSON.
4. Compartir el CSV de Drive con el correo de la Service Account (permiso lector).
5. Configurar `.env` con `WYSCOUT_DRIVE_FILE_ID` y `GOOGLE_SERVICE_ACCOUNT_FILE`.

## Ejecución
```bash
streamlit run app.py
```

## Despliegue en Streamlit Cloud
- Subir el repo a GitHub.
- En Streamlit Cloud, crear app apuntando a `app.py`.
- Definir `Secrets` con tus variables de entorno (`WYSCOUT_DRIVE_FILE_ID`, `GOOGLE_SERVICE_ACCOUNT_JSON`, etc.).
- No subir claves privadas al repositorio.

## Limpieza de repositorio
El proyecto ignora artefactos generados y exportaciones pesadas mediante `.gitignore`:
- `pdf_exports/`, `common/pdf_exports/`
- `figures/pizzas/`, `figures/scatterplots/`, `figures/tables/`
- imágenes temporales `.jpg/.jpeg/.zip` en `figures/images/`

## Seguridad
- No commitear `.env` ni claves JSON.
- Rotar credenciales si se exponen por error.
- Mantener permisos mínimos en Drive/API.

## Créditos
Desarrollada por **Ramón Codesido**  
Contacto: `ramon.codesido@gmail.com`
