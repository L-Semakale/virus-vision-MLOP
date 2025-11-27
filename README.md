# **Virus Vision — MLOps Pipeline for Virus Image Classification**

Virus Vision is an end-to-end Machine Learning and MLOps project designed to classify virus images using deep learning.
The project follows industry-grade best practices, including modular code architecture, model training pipelines, experiment tracking, CI/CD, containerization, and deployment.

---

##  **Project Overview**

This project implements a fully-structured Machine Learning lifecycle:

* **Data ingestion & preprocessing**
* **Model training & optimization**
* **Evaluation & experiment tracking**
* **Packaging the model**
* **Serving the model through an API**
* **Containerization via Docker**
* **CI/CD integration (GitHub Actions)**
* **Scalable deployment**

The goal is to classify virus images accurately using a well-maintained, reproducible workflow.

---

##  **Project Structure**

```
virus-vision-MLOP/
│
├── data/                     # Raw & processed dataset (ignored in repo)
├── notebooks/                # Experimentation & exploratory analysis
├── src/
│   ├── data/                 # Data ingestion scripts
│   ├── models/               # Model architectures & saved checkpoints
│   ├── training/             # Training loops & evaluation
│   ├── api/                  # FastAPI/Flask model serving
│   └── utils/                # Helpers & common functions
│
├── Dockerfile                # Container build file
├── requirements.txt          # Project dependencies
├── README.md                 # Documentation
└── .github/workflows/        # CI/CD pipelines
```

---

##  **Model**

* **Architecture:** CNN or transfer learning (ResNet/EfficientNet)
* **Task:** Multi-class virus image classification
* **Framework:** PyTorch
* **Trained model saved as:** `model.pt`

---

##  **Setup & Installation**

### 1. Clone the repository

```bash
git clone https://github.com/L-Semakale/virus-vision-MLOP.git
cd virus-vision-MLOP
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux & Mac
.\.venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  **Training the Model**

```bash
python src/training/train.py --config configs/train_config.yaml
```

Model checkpoints will be stored inside:

```
src/models/checkpoints/
```

---

##  **Inference / Making Predictions**

Once you have `model.pt`, run:

```bash
python src/api/predict.py --image path/to/image.jpg
```

---

##  **API Usage (FastAPI)**

Start the API:

```bash
uvicorn src.api.main:app --reload
```

Send a prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@virus_image.jpg"
```

---

##  **Docker Deployment**

Build the image:

```bash
docker build -t virus-vision .
```

Run the container:

```bash
docker run -p 8000:8000 virus-vision
```

---

##  **CI/CD Pipeline**

The project includes:

* automated linting
* automated testing
* model packaging
* API build & deployment

All handled through GitHub Actions.

---

##  **Demo Video**

(Your video link will be added here)

---
