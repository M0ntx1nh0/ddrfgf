import streamlit as st


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-main: #f5f7fb;
                --bg-card: #ffffff;
                --text-main: #1f2937;
                --accent: #0f4c81;
                --accent-2: #00a676;
                --border-soft: #dce3ee;
            }

            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, var(--bg-main) 100%);
                color: var(--text-main);
            }

            [data-testid="stSidebar"] {
                background: #f0f4fa;
                border-right: 1px solid var(--border-soft);
            }

            [data-testid="stHeader"] {
                background: rgba(245, 247, 251, 0.75);
            }

            [data-testid="stTabs"] [role="tab"] {
                border: 1px solid var(--border-soft);
                border-radius: 10px 10px 0 0;
                background: #eef3fa;
                color: #35516b;
                padding: 8px 14px;
            }

            [data-testid="stTabs"] [aria-selected="true"] {
                background: var(--bg-card);
                color: var(--accent);
                border-bottom: 2px solid var(--accent);
                font-weight: 700;
            }

            .stButton > button {
                background: var(--accent);
                color: #ffffff;
                border: 1px solid var(--accent);
                border-radius: 8px;
            }

            .stButton > button:hover {
                background: #0b3e68;
                border-color: #0b3e68;
            }

            [data-testid="stDataFrame"] div[role="table"] {
                border: 1px solid var(--border-soft);
                border-radius: 10px;
                overflow: hidden;
            }

            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3 {
                color: var(--accent);
                letter-spacing: 0.2px;
            }

            .stExpander {
                border: 1px solid var(--border-soft);
                border-radius: 10px;
                background: var(--bg-card);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
