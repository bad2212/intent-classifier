# CPU serving image for the intent router (model pulled from the Hugging Face Hub at start-up).
#   docker build -t intent-router .
#   docker run -p 8000:8000 -e MODEL_ID=Badalt/intent-classifier-mpnet intent-router
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==5.18.0 fastapi==0.142.2 uvicorn==0.54.0 sentencepiece
COPY src/__init__.py src/predict.py src/serve.py src/
ENV DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
