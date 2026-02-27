from pathlib import Path

import streamlit as st
from PIL import Image


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
