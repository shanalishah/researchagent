import streamlit as st

from src.ui import apply_branding, render_footer


st.set_page_config(
    page_title="Settings",
    page_icon="🔐",
    layout="centered",
)

apply_branding()

st.title("Settings")

st.write(
    """
    This page explains how API keys and backend settings should be handled for the
    public deployment.
    """
)

st.subheader("Recommended public setup")

st.markdown(
    """
    - Default mode: **Free Local**
    - Optional advanced modes: **OpenAI**, **Gemini**, or **Groq**
    - API keys should never be committed to GitHub.
    - Store deployment secrets in Streamlit Cloud Secrets.
    """
)

st.subheader("Streamlit Secrets format")

st.code(
    """
OPENAI_API_KEY = "your_openai_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
GROQ_API_KEY = "your_groq_key_here"

# Optional later if you reconnect Cloudflare R2:
R2_ACCESS_KEY_ID = "your_r2_access_key"
R2_SECRET_ACCESS_KEY = "your_r2_secret_key"
R2_ENDPOINT = "your_r2_endpoint"
R2_BUCKET = "your_r2_bucket"
""".strip(),
    language="toml",
)

st.subheader("Current status")

has_openai = bool(st.secrets.get("OPENAI_API_KEY", ""))
has_gemini = bool(st.secrets.get("GEMINI_API_KEY", ""))
has_groq = bool(st.secrets.get("GROQ_API_KEY", ""))

status_rows = [
    ("OpenAI key configured", has_openai),
    ("Gemini key configured", has_gemini),
    ("Groq key configured", has_groq),
]

for label, status in status_rows:
    if status:
        st.success(f"{label}: yes")
    else:
        st.info(f"{label}: not configured")

st.warning(
    """
    This starter version does not send user queries to OpenAI, Gemini, or Groq yet.
    The provider dropdown is included so the UI is ready for the next backend connection step.
    """
)

render_footer()
