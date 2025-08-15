# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Add an argument for the model path from the workflow
ARG MODEL_PATH
# Copy the trained model from the specific path to a fixed location
COPY ${MODEL_PATH} /model/

COPY . /app/
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]