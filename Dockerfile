FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "episteward.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
