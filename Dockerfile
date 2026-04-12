FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn numpy pydantic

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
