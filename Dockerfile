# -slim-bookworm
FROM python:3.10

WORKDIR /app

COPY app/ /app/
COPY requirements.txt /app/requirements.txt
COPY data/gcp_service.json /app/data/gcp_service.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/data/gcp_service.json"

RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
