FROM python:3.11-slim

# Install Node.js + npm for building frontend
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build frontend first
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install && npm run build

# Install Python deps
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend
COPY frontend/dist/ ./frontend/dist/

ENV PORT=7860
EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
