FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model from the mlruns directory using a wildcard
# This finds the model regardless of the exact run ID
COPY mlruns/0/*/artifacts/model/ /model/

COPY . /app/
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]