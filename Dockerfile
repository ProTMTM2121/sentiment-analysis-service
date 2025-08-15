FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model from the simple, known location
COPY ./outputs/model.joblib /model/model.joblib

COPY . /app/
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]