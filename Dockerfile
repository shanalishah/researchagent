# Research Agent — single-container deploy (React frontend + FastAPI backend).
#
# Stage 1 builds the Vite/React app to static files.
# Stage 2 is the Python backend, which serves those static files AND the API
# from one process — so the whole app is one Cloud Run service.
#
# (The previous Streamlit container is preserved as Dockerfile.streamlit.)

# ---- Stage 1: build the React frontend ----
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend serving API + built frontend ----
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Install CPU-only PyTorch first (keeps the image small — the default torch
# wheel bundles multi-GB CUDA libraries we don't need on Cloud Run).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# App code + the built frontend from stage 1
COPY backend/ ./backend/
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8080

# Bind to 0.0.0.0 and the Cloud Run-provided PORT.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
