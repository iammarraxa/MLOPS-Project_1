# 🚗 Vehicle Data Prediction — End-to-End MLOps Project

### 🔥 *An intelligent, fully-automated ML system built from scratch with CI/CD, Docker, AWS, and scalable cloud infrastructure.*

---

## 🌟 Project Overview

This project demonstrates how to take a **machine learning model from development to deployment** using modern **MLOps practices**.
It covers everything — data ingestion, validation, transformation, model training, evaluation, and automated deployment using AWS and GitHub Actions.

> **Goal:** Build a production-ready machine learning pipeline for vehicle data analysis and prediction, fully automated through CI/CD on AWS infrastructure.

---

## 🧠 Tech Stack & Tools

| Category                   | Tools / Services Used                           |
| -------------------------- | ----------------------------------------------- |
| **Language & Environment** | Python 3.10, Conda, Jupyter Notebook            |
| **Data Storage**           | MongoDB Atlas                                   |
| **ML & Data Handling**     | pandas, NumPy, scikit-learn, PyYAML             |
| **MLOps & Cloud**          | AWS (S3, EC2, IAM, ECR)                         |
| **CI/CD**                  | GitHub Actions (self-hosted runner on EC2)      |
| **Containerization**       | Docker                                          |
| **Logging & Monitoring**   | Python `logging` module                         |
| **Web Framework**          | Flask (for web UI + inference API)              |
| **Version Control**        | Git & GitHub                                    |
| **Deployment**             | AWS EC2 with Nginx proxy (optional HTTPS setup) |

---

## 🧩 System Architecture

```
                ┌──────────────────────────────┐
                │        Data Source           │
                └────────────┬─────────────────┘
                             │
                             ▼
               ┌──────────────────────────────┐
               │     Data Ingestion (Mongo)    │
               └──────────────────────────────┘
                             │
                             ▼
               ┌──────────────────────────────┐
               │ Data Validation & Transform   │
               └──────────────────────────────┘
                             │
                             ▼
               ┌──────────────────────────────┐
               │    Model Training & Eval      │
               └──────────────────────────────┘
                             │
                             ▼
               ┌──────────────────────────────┐
               │   Model Registry (AWS S3)     │
               └──────────────────────────────┘
                             │
                             ▼
               ┌──────────────────────────────┐
               │     Flask API / Web App       │
               └──────────────────────────────┘
```

---

## ⚙️ Setup & Installation

### 1️⃣ Create Project Template

```bash
python template.py
```

### 2️⃣ Setup Local Packages

Configure `setup.py` and `pyproject.toml` to handle local imports.

> Learn more: See `crashcourse.txt` for quick setup reference.

### 3️⃣ Create Virtual Environment

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

Verify packages:

```bash
pip list
```

---

## ☁️ MongoDB Atlas Integration

1. **Sign up** at [MongoDB Atlas](https://www.mongodb.com/atlas).
2. **Create a new cluster** (Free Tier M0).
3. **Add a database user** and whitelist IP `0.0.0.0/0`.
4. **Copy connection string** (Python Driver).
5. Push your dataset from Jupyter notebook (`mongoDB_demo.ipynb`).
6. Verify your collection inside the MongoDB console.

---

## 🧾 Logging & Exception Handling

* Implemented structured logging via `src/logger`
* Centralized error tracking via `src/exception`
* Verified with `demo.py`

---

## 📊 Data Pipeline Components

### **1. Data Ingestion**

* Connects to MongoDB
* Fetches raw data
* Converts key–value pairs → Pandas DataFrame
* Saves intermediate artifacts

### **2. Data Validation**

* Checks schema consistency via `schema.yaml`
* Detects missing, null, or drifted data

### **3. Data Transformation**

* Performs scaling, encoding, and feature engineering
* Saves transformed data for reproducibility

### **4. Model Trainer**

* Trains multiple algorithms
* Tracks performance metrics
* Selects best model based on threshold

### **5. Model Evaluation & Pusher**

* Compares current and previous models
* Pushes best model to AWS S3 (`my-model-mlopsproj/model-registry/`)

---

## ☁️ AWS Integration

* **S3** — Model registry storage
* **ECR** — Container image registry
* **EC2** — Hosting environment (self-hosted GitHub runner)
* **IAM** — Secure key-based access

**IAM Setup (Quick View):**

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-1"
```

Bucket naming convention:

```
MODEL_BUCKET_NAME = "my-model-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"
```

---

## 🐳 CI/CD & Deployment

### ✅ CI (Continuous Integration)

* Triggered on push to `main`
* Builds and tags Docker image
* Pushes image to AWS ECR

### 🚀 CD (Continuous Deployment)

* Self-hosted runner on EC2 pulls the latest image
* Stops old container
* Runs new one automatically
* Exposes app on port `5000`

### GitHub Secrets

| Key                     | Description                 |
| ----------------------- | --------------------------- |
| `AWS_ACCESS_KEY_ID`     | AWS Access Key              |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key              |
| `AWS_DEFAULT_REGION`    | AWS Region                  |
| `ECR_REPO`              | Name of your ECR repository |

---

## 🌐 Running the App

Once the CI/CD pipeline finishes:

1. Visit your EC2 instance’s public IP

   ```bash
   http://<ec2-public-ip>:5000/
   ```
2. Flask web interface will load.
3. For model retraining:

   ```bash
   http://<ec2-public-ip>:5000/training
   ```

---

## 🛠️ Project Highlights

✔️ **Modular Code Structure** — reusable, testable, and production-ready
✔️ **Automated CI/CD Pipeline** — from commit → deployment in under a minute
✔️ **Cloud-Native Architecture** — all infrastructure built on AWS
✔️ **Data-Driven Workflow** — MongoDB Atlas + Pandas + PyYAML schema
✔️ **Scalable Deployment** — Dockerized app on EC2 with self-hosted runner
✔️ **Fail-Safe Logging** — every step traceable through logs

---

## 🧭 Directory Structure

```
├── .github/workflows/aws.yaml
├── src/
│   ├── components/
│   ├── configuration/
│   ├── data_access/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   └── pipeline/
├── notebook/
│   ├── EDA_FeatureEngg.ipynb
│   └── mongoDB_demo.ipynb
├── template.py
├── app.py
├── requirements.txt
├── Dockerfile
├── setup.py
├── pyproject.toml
└── README.md
```

---

## 🧪 Example Workflow Summary

| Step | Description                                           |
| ---- | ----------------------------------------------------- |
| 1    | Data fetched from MongoDB Atlas                       |
| 2    | Validation, transformation & feature engineering      |
| 3    | Model trained and saved as artifact                   |
| 4    | Model pushed to AWS S3                                |
| 5    | Docker image built & pushed to ECR                    |
| 6    | EC2 auto-deploys containerized app via GitHub Actions |

---

## 🧾 Routes

| Endpoint    | Description                      |
| ----------- | -------------------------------- |
| `/`         | Home Page                        |
| `/training` | Triggers model training pipeline |

---

## 🧩 Future Enhancements

* Add monitoring via **Prometheus + Grafana**
* Integrate **Airflow / Kubeflow** for orchestration
* Enable **HTTPS (NGINX + Let’s Encrypt)**
* Extend multi-model support for different vehicle classes

---

## 🧑‍💻 Author

**Muhammad Ammar Raza**
🌐 [LinkedIn](https://www.linkedin.com/in/muhammad-ammar-raza/) | 🧠 MLOps | Cloud | AI

---

## 💬 Final Note

> This project is not just an ML model — it’s a demonstration of complete **MLOps lifecycle automation**, from data to deployment.
> It showcases how real-world systems are built, automated, and scaled using modern cloud-native tools.


