FROM python:3.11-slim

# ------------------------------------------------------------------ system deps
# matplotlib needs libfreetype + libpng; clean up apt cache to keep image small
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libfreetype6-dev \
        libpng-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ python env
WORKDIR /app

# Prevent pip from buffering stdout (better Docker logs)
ENV PYTHONUNBUFFERED=1
# Don't write .pyc files inside the container
ENV PYTHONDONTWRITEBYTECODE=1

# ------------------------------------------------------------------ dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------ application
COPY . .

# Create artifacts dir so plotting_tool doesn't fail on first run
RUN mkdir -p artifacts

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
