# ======== Base Image ========
FROM python:3.10-slim

# ======== Set Workdir ========
WORKDIR /app

# ======== Install System Dependencies ========
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ======== Copy Requirements ========
COPY requirements.txt ./

# ======== Install Python Dependencies ========
RUN pip install --no-cache-dir -r requirements.txt

# ======== Copy Project Files ========
COPY . .

# ======== Expose API Port ========
EXPOSE 8000

# ======== Start FastAPI Server ========
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
