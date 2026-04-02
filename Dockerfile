FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    chromium \
    && rm -rf /var/lib/apt/lists/*

ENV BROWSER=/usr/bin/chromium
ENV CHROME_PATH=/usr/bin/chromium

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data
