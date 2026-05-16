FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PREFECT_API_URL=http://prefect:4200/api

WORKDIR /app

# build-essential : compilation des wheels C
# libhdf5-dev    : requis par h5py (dépendance de dice-ml 0.9)
# curl, git      : outils MLOps (mlflow, prefect)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["tail", "-f", "/dev/null"]