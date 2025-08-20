FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements/host_requirements.txt .
RUN pip install --no-cache-dir -r host_requirements.txt
COPY host /host
COPY data /data

CMD ["python", "-m", "host.mcp_host"]