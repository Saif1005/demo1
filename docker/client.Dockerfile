FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements/client_requirements.txt .
RUN pip install --no-cache-dir -r client_requirements.txt
COPY clients /clients
COPY data /data

CMD ["python", "-m", "clients.flower_client"]