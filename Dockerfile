FROM python:3.10-slim

WORKDIR /app

COPY requirement.txt .
RUN apt-get update && apt-get install -y gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--log-level", "debug", "-b", "0.0.0.0:8000", "app:app"]