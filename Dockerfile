FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.4 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    python-dotenv==1.0.1

# Copy app code and models
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run
CMD ["python", "src/api/main.py"]