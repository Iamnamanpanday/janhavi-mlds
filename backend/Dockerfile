FROM python:3.12-slim
RUN useradd --system --create-home scanner
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
RUN python -m app.train --smoke
RUN mkdir /app/data && chown -R scanner:scanner /app/data
USER scanner
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
