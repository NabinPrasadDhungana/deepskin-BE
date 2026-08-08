# ModelScope Studios container for the DeepSkin Django backend.
# Thin image: ML inference runs remotely (see DEEPSKIN_HF_URL), so the
# TensorFlow model is deliberately NOT bundled here. See mlservice/pipeline.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# psycopg2-binary is prebuilt, but keep build-essential as a safety net for
# any pip wheel that needs compilation in a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /code

RUN chmod +x /code/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/code/entrypoint.sh"]