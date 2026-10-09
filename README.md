# Credit Approval System & AI Banking Assistant

An AI-powered credit decision-support web application combining **Machine Learning** and **Retrieval-Augmented Generation (RAG)** to support banking and loan-related tasks.

The application uses a **Random Forest classifier** to predict credit approval outcomes and assess applicant risk. It also integrates a conversational AI assistant that retrieves relevant information from a banking FAQ knowledge base and generates context-aware answers.

Built with **Python, FastAPI, scikit-learn, ChromaDB, Sentence Transformers, and the Groq API**, this project combines ML inference, REST API development, software design principles, and generative AI in a single application.

> Academic team project developed as part of the Master's program in Artificial Intelligence and Data Analytics at Université Ibn Zohr.

---

## Features

### Machine Learning Credit Assessment

* Predicts **APPROVED / REJECTED** outcomes from applicant financial and credit information.
* Uses a trained Random Forest classification model.
* Returns a prediction confidence score and risk level.
* Provides model information and health-check endpoints.
* Validates incoming requests using Pydantic.
* Logs prediction events using FastAPI background tasks.

### AI Banking Assistant

* Conversational assistant integrated into the credit approval application.
* Uses **Retrieval-Augmented Generation (RAG)** to answer questions using a banking FAQ knowledge base.
* Retrieves relevant documents from a ChromaDB vector database.
* Generates answers using the Groq API and Llama 3.1.
* Uses Sentence Transformers (`all-MiniLM-L6-v2`) to convert text into embeddings.
* Supports a fallback response when the available context is insufficient.

### Software Engineering

* REST API built with FastAPI.
* Dependency Inversion Principle and interface-based architecture.
* Dependency injection and separation of responsibilities.
* Background logging for prediction requests.
* Browser-based interface served directly by FastAPI.

---

## How It Works

### 1. Credit Assessment

1. The user submits applicant information through the web interface.
2. Pydantic validates the request data.
3. The application preprocesses the input using the configured data-processing pipeline.
4. The Random Forest model generates a prediction and approval probability.
5. The application returns the decision, confidence score, and risk level.
6. Prediction events are logged using FastAPI background tasks.

### 2. AI Banking Assistant

1. The user submits a question through the chat interface.
2. The frontend sends the question to the FastAPI `/api/chat` endpoint.
3. Sentence Transformers converts the question into an embedding.
4. ChromaDB retrieves relevant information from the banking FAQ knowledge base.
5. The retrieved context and question are sent to the Groq API using Llama 3.1.
6. The generated answer is returned to the frontend.

This approach combines information retrieval with language generation to provide answers grounded in the project's banking knowledge base.

---

## Tech Stack

| Technology            | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| Python                | Main programming language                         |
| scikit-learn          | Random Forest classification and model evaluation |
| FastAPI               | REST API and web application backend              |
| Pydantic              | Request and response validation                   |
| Pandas                | Data loading and preprocessing                    |
| NumPy                 | Numerical operations                              |
| Uvicorn               | ASGI server                                       |
| Groq API              | Language model access                             |
| Llama 3.1             | Conversational answer generation                  |
| Sentence Transformers | Text embeddings                                   |
| `all-MiniLM-L6-v2`    | Embedding model producing 384-dimensional vectors |
| ChromaDB              | Vector storage and similarity-based retrieval     |
| HTML, CSS, JavaScript | Web interface and chatbot interactions            |

---

## Architecture

The application separates API handling, ML processing, model persistence, and chatbot functionality into dedicated components.

```text
credit_approval_system/
├── api/
│   ├── main.py              # FastAPI endpoints and web interface
│   └── schemas.py           # Request and response schemas
├── chatbot/
│   ├── chatbot.py           # RAG retrieval and answer generation
│   ├── ingest.py            # FAQ ingestion and vector storage
│   └── app.py               # Chatbot interface/testing
├── data/
│   ├── raw/
│   │   └── credit_data.csv  # Credit applicant dataset
│   └── bank_faq/
│       └── bank_faq.txt     # Banking FAQ knowledge base
├── models/
│   └── credit_model_info.json
├── src/
│   ├── interfaces.py        # Abstract interfaces
│   ├── data_loader.py       # Data preprocessing
│   ├── model.py             # Model training and evaluation
│   ├── predictor.py         # Prediction orchestration
│   ├── repository.py        # Model persistence
│   ├── decorators.py        # Logging, timing and validation
│   ├── functional.py        # Functional programming utilities
│   ├── logger.py            # Logging functionality
│   ├── container.py         # Dependency injection container
│   └── utils.py             # Utility functions
├── train.py                 # Model training entry point
├── tests.py                 # Project tests
├── requirements.txt         # Python dependencies
└── README.md
```

Generated artifacts, including the trained `.pkl` model and ChromaDB's persistent vector store, are excluded from version control and should be generated or initialized locally when required.

### Design Principles and Patterns

* **Dependency Inversion Principle (DIP):** components depend on abstractions rather than concrete implementations.
* **Abstract Base Classes:** define contracts for model, preprocessing, prediction, repository, and logging components.
* **Dependency Injection:** manages component dependencies and improves testability.
* **Singleton:** provides a shared dependency injection container.
* **Decorators:** support cross-cutting concerns such as timing and execution logging.
* **Background Tasks:** handle prediction logging without blocking the response on the logging operation.

---

## Machine Learning Model

### Dataset

The current training pipeline processes **1,000 applicant records**, using 12 input features and one binary target: `Approval`.

The input features include:

* Age
* Monthly income
* Employment status and duration
* Total debt
* Loan amount and duration
* Credit score
* Repayment history
* Number of late payments
* Number of open accounts
* Account age in years

### Training Pipeline

The pipeline loads and preprocesses the dataset, separates the features from the target, splits the data into training and testing sets, trains the Random Forest classifier, and evaluates its performance on held-out test data.

The latest recorded training run used:

* Training samples: 800
* Testing samples: 200
* Model: Random Forest Classifier
* Decision threshold: approval probability of at least 0.5

### Evaluation Results

The following metrics were obtained from the latest recorded training run:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 84.00% |
| Precision | 85.51% |
| Recall    | 90.77% |
| F1 Score  | 88.06% |
| ROC-AUC   | 92.73% |

These results describe performance on the project's test split. They do not guarantee equivalent performance on real banking data. This model is an academic demonstration and should not be used to make real lending decisions without appropriate validation, fairness assessment, security controls, and regulatory review.

---

## Retrieval-Augmented Generation (RAG)

The chatbot demonstrates how document retrieval and a large language model can be combined to answer questions using a domain-specific knowledge base.

### Components

| Component             | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| Groq API              | Provides access to the language model                   |
| Llama 3.1             | Generates conversational answers                        |
| Sentence Transformers | Converts questions and documents into embeddings        |
| `all-MiniLM-L6-v2`    | Produces 384-dimensional text embeddings                |
| ChromaDB              | Stores embeddings and retrieves relevant documents      |
| `bank_faq.txt`        | Contains banking and loan-related questions and answers |
| FastAPI               | Exposes the chatbot through the application API         |

### RAG Pipeline

```text
User Question
     |
     v
FastAPI /api/chat
     |
     v
Question Embedding
     |
     v
ChromaDB Retrieval
     |
     v
Relevant FAQ Context
     |
     v
Groq API - Llama 3.1
     |
     v
Generated Answer
     |
     v
Chat Interface
```

The FAQ knowledge base contains 24 question-and-answer entries. The ingestion script processes the knowledge base and stores its embeddings in the local ChromaDB vector store.

---

## API Reference

| Method | Endpoint          | Description                                     |
| ------ | ----------------- | ----------------------------------------------- |
| `GET`  | `/`               | Serves the web interface                        |
| `GET`  | `/api/health`     | Checks API health and model status              |
| `GET`  | `/api/model-info` | Returns model information and training metadata |
| `POST` | `/api/predict`    | Predicts a credit approval outcome              |
| `POST` | `/api/chat`       | Sends a question to the RAG banking assistant   |

Interactive API documentation is available through Swagger UI at `/docs` when the server is running.

---

## Getting Started

### Prerequisites

* Python 3.11 recommended
* Git
* A Groq API key for chatbot functionality

### 1. Clone the Repository

```bash
git clone https://github.com/wissalkabous/credit_approval_system.git
cd credit_approval_system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the chatbot dependencies used by the project, including `groq`, `chromadb`, and `sentence-transformers`.

### 4. Configure the Groq API Key

Create a `.env` file in the project root and add your own API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or expose API keys in source code or public repositories.

### 5. Train the Credit Model

```bash
python train.py
```

This generates the trained model and associated metadata locally.

### 6. Initialize the Chatbot Knowledge Base

```bash
python chatbot/ingest.py
```

This processes the banking FAQ knowledge base and initializes the local ChromaDB vector store. Run ingestion again when the knowledge base changes, if required by the implementation.

### 7. Start the API

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open the following URLs in your browser:

* Web application: http://localhost:8000
* Swagger API documentation: http://localhost:8000/docs
* API health check: http://localhost:8000/api/health

The chatbot requires valid Groq API configuration and an initialized local vector store.

---

## Future Improvements

* Add automated tests for API endpoints, prediction behavior, and RAG retrieval.
* Improve retrieval quality and evaluate chatbot answers against a test set.
* Add conversation history and source references for retrieved answers.
* Improve model validation using cross-validation and hyperparameter tuning.
* Add authentication, authorization, and stronger security controls.
* Add monitoring and structured evaluation for model and chatbot performance.

---

## Team

Developed as a team project for the Machine Learning and Python Programming modules of the Master's program in Artificial Intelligence and Data Analytics at Université Ibn Zohr.

## License

This project was developed for academic purposes and does not currently specify a formal open-source license.
