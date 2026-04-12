FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn numpy

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
