# SSCA Backend & Model AI Mindspore

A comprehensive Flask-based backend system for analyzing and optimizing supply chain sustainability. This project provides AI-powered solutions for emission forecasting, supplier risk assessment, regulatory compliance analysis, and supply chain optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Models & Components](#models--components)
- [Data Files](#data-files)
- [Development](#development)

## 🌍 Overview

SSCA (Supply Chain Carbon Analytics) Backend is designed to help organizations understand, analyze, and reduce their carbon footprint across the entire supply chain. It combines machine learning models with advanced analytics to provide actionable insights for sustainability improvements.

### Key Capabilities:

- **Emission Forecasting**: LSTM-Attention based models to predict future emissions
- **Supplier Risk Assessment**: Multi-dimensional risk scoring for supplier evaluation
- **Regulatory Compliance**: NLP-powered analysis of regulatory documents and requirements
- **Supply Chain Optimization**: Multi-objective optimization for sustainable scenarios
- **ESG Analysis**: Comprehensive Environmental, Social, and Governance evaluation
- **Report Generation**: Automated sustainability and compliance reporting

## ✨ Features

### 1. Emission Forecasting
- Predicts future carbon emissions using deep learning models
- Processes 30 days of historical operational data
- Supports multivariate input features (production volume, energy consumption, shipments, etc.)
- Returns forecasts with confidence intervals

### 2. Supplier Risk Scoring
- Evaluates supplier sustainability risk across four dimensions:
  - Environmental Impact (40% weight)
  - Compliance Status (30% weight)
  - Operational Resilience (20% weight)
  - Reputational Risk (10% weight)
- Provides composite risk scores (0-100)
- Batch processing capabilities

### 3. Regulatory Compliance Analysis
- Extracts key regulatory information from documents using NLP
- Identifies:
  - Threshold values and applicability criteria
  - Compliance deadlines
  - Penalties and enforcement mechanisms
  - Reporting scopes
  - Emission targets
  - Renewable energy requirements
- Conversational AI for regulatory queries

### 4. Supply Chain Optimization
- Multi-objective optimization across cost and emissions
- Scenario analysis with modal shift and renewable energy adjustments
- Supports:
  - Modal shift (air/truck → rail/sea)
  - Renewable energy integration
  - Cost-benefit analysis

### 5. Report Generation
- Automated generation of:
  - Sustainability reports
  - Compliance assessments
  - Executive summaries
  - Analytics dashboards

## 📁 Project Structure

```
backend/
├── app.py                          # Main Flask application
├── Dockerfile                      # Docker container configuration
├── docker-compose.yaml             # Docker Compose setup
├── requirement.txt                 # Python dependencies
├── README.md                       # This file
│
├── models/                         # Machine learning models
│   ├── emission_forecasting.py    # LSTM-Attention emission prediction
│   ├── optimization.py            # Multi-objective supply chain optimization
│   ├── regulatory_nlp.py          # NLP for regulatory document analysis
│   ├── supplier_risk.py           # Risk scoring models
│   └── __pycache__/               # Compiled Python files
│
├── utils/                          # Utility modules
│   ├── data_processor.py          # Data processing and validation
│   ├── validators.py              # Input validation functions
│   └── __pycache__/               # Compiled Python files
│
├── data/                           # Data files and resources
│   ├── emission_factors.json      # CO2 emission factors by transport mode
│   ├── scenarios.json             # Predefined supply chain scenarios
│   ├── sample_logistics_routes.csv # Sample shipping routes data
│   ├── sample_operational_logs.csv # Sample operational metrics
│   └── sample_supplier_esg.csv    # Sample supplier ESG data
│
└── save_models/                    # Trained model storage
```

## 🛠 Technology Stack

### Backend Framework
- **Flask 3.1.2** - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Python 3.10** - Programming language

### Machine Learning & AI
- **MindSpore** - Deep learning framework
- **Scikit-learn 1.7.2** - Machine learning algorithms
- **TensorFlow/Transformers** - NLP models
- **Pandas 2.3.3** - Data manipulation
- **NumPy 1.26.4** - Numerical computing
- **SciPy 1.15.3** - Scientific computing

### Data & Visualization
- **Matplotlib 3.10.7** - Data visualization
- **Pillow 12.0.0** - Image processing

### Document Processing
- **python-docx 1.2.0** - Word document generation
- **openpyxl 3.1.5** - Excel file handling
- **reportlab 4.4.5** - PDF generation

### Additional Libraries
- **requests 2.32.5** - HTTP client
- **python-dotenv 1.2.1** - Environment variables
- **regex 2025.11.3** - Advanced text processing

## 💻 Installation & Setup

### Prerequisites
- Docker & Docker Compose (recommended)
- Python 3.10+
- pip/conda package manager

### Option 1: Docker Setup (Recommended)

1. **Clone the repository**
   ```bash
   cd /home/dauzan/ssca/backend
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

The service will be available at `http://localhost:5000`

### Option 2: Local Python Setup

1. **Create virtual environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

The application will start on `http://localhost:5000`

## 🚀 Running the Application

### Start the Server

**Using Docker Compose:**
```bash
docker-compose up
```

**Using Python directly:**
```bash
python app.py
```

### Verify Installation

Check the health endpoint:
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-10T..."
}
```

## 📚 API Documentation

### Base URL
```
http://localhost:5000/api/v1
```

### Core Endpoints

#### 1. Emission Forecasting
- **POST** `/forecast/emissions`
  - Predicts future carbon emissions
  - Input: Historical operational data (30 days)
  - Output: Emission forecasts with confidence intervals

#### 2. Supplier Risk Scoring
- **POST** `/score/supplier`
  - Evaluates supplier sustainability risk
  - Input: Supplier ESG metrics
  - Output: Risk score (0-100) with component breakdown

#### 3. Supply Chain Optimization
- **POST** `/optimize/supply-chain`
  - Optimizes scenarios for cost and emissions
  - Input: Scenario parameters (modal shift %, renewable energy %)
  - Output: Optimized configuration and savings

#### 4. Regulatory Analysis
- **POST** `/analyze/regulation`
  - Analyzes single regulatory document
  - Input: Regulatory text
  - Output: Extracted entities and compliance requirements

- **POST** `/batch/analyze/regulations`
  - Batch processes multiple regulations
  - Input: List of regulatory texts
  - Output: Analysis results with compliance mapping

#### 5. Conversational Regulatory AI
- **POST** `/chat/regulation`
  - Interactive Q&A for regulatory queries
  - Input: Question about regulations
  - Output: Contextual answer with sources

#### 6. Report Generation
- **POST** `/generate/report`
  - Generates sustainability reports
  - Input: Report parameters and data
  - Output: Report ID for status tracking

- **GET** `/report/status/<report_id>`
  - Checks report generation status
  - Output: Current status and progress

- **GET** `/download/<report_id>`
  - Downloads generated report
  - Output: PDF or Excel file

#### 7. Supplier Management
- **GET** `/suppliers`
  - List all suppliers
  - Output: Array of supplier records

- **POST** `/suppliers`
  - Create new supplier
  - Input: Supplier information
  - Output: Created supplier with ID

- **PUT** `/suppliers/<supplier_id>`
  - Update supplier information
  - Input: Updated supplier data
  - Output: Updated supplier record

- **DELETE** `/suppliers/<supplier_id>`
  - Remove supplier
  - Output: Confirmation message

- **POST** `/suppliers/upload`
  - Bulk upload suppliers
  - Input: CSV file with supplier data
  - Output: Upload status and results

#### 8. Scenario Management
- **GET** `/scenarios`
  - List available scenarios
  - Output: Array of scenario definitions

- **POST** `/scenarios`
  - Create new scenario
  - Input: Scenario configuration
  - Output: Created scenario with ID

#### 9. Analytics
- **GET** `/suppliers/analytics`
  - Comprehensive supplier analytics dashboard
  - Output: Aggregated metrics, trends, and insights

### Health Check
- **GET** `/health`
  - System health status
  - Output: Status and timestamp

- **GET** `/`
  - Root endpoint with API information
  - Output: API overview and documentation

## 🧠 Models & Components

### 1. **EmissionForecasting** (`models/emission_forecasting.py`)
- **Type**: LSTM-Attention neural network
- **Input Features**: 11 multivariate dimensions
  - Production volume
  - Energy consumption (kWh)
  - Gas consumption (m³)
  - Runtime hours
  - Shipment count
  - Distance traveled
  - Modal split percentages
  - And more operational metrics
- **Architecture**:
  - 2 LSTM layers (64 hidden units)
  - Attention mechanism for feature importance
  - Sequence length: 30 days of historical data
  - Output: CO2e emission predictions

### 2. **SupplierRiskScorer** (`models/supplier_risk.py`)
- **Type**: Multi-layer neural network
- **Input Features**: 15 ESG-related dimensions
- **Risk Dimensions**:
  - Environmental (40% weight) - emissions, waste, water usage
  - Compliance (30% weight) - certifications, violations, audits
  - Operational (20% weight) - reliability, capacity, disruptions
  - Reputational (10% weight) - controversies, media coverage
- **Output**: Risk score (0-100)

### 3. **RegulatoryNLP** (`models/regulatory_nlp.py`)
- **Type**: Transformer-based Named Entity Recognition (NER)
- **Architecture**:
  - Transformer encoder (6 layers, 12 attention heads)
  - Word embeddings (30,522 vocab size)
  - Maximum sequence length: 512 tokens
- **Entities Extracted** (7 types):
  - THRESHOLD - Applicability criteria (e.g., "750 employees")
  - DEADLINE - Compliance dates
  - PENALTY - Enforcement actions
  - SCOPE - Reporting boundaries
  - ENTITY - Covered organizations
  - EMISSION_VALUE - Reported emission numbers
  - RENEWABLE_PCT - Green energy targets
- **Features**:
  - Batch document processing
  - Multi-language support
  - Regulatory compliance mapping

### 4. **ScenarioOptimizer** (`models/optimization.py`)
- **Type**: Multi-objective optimization using genetic algorithms
- **Parameters**:
  - Modal shift percentage (air/truck → rail/sea)
  - Renewable energy increase percentage
- **Optimization**:
  - Emission reduction targets
  - Cost-benefit analysis
  - Multi-scenario comparison
- **Baseline**:
  - Annual emissions: 100,000 kg CO2e
  - Baseline cost: IDR 8,337,500,000

### 5. **SupplierESGAnalyzer** (`models/regulatory_nlp.py`)
- **Purpose**: Extract and score ESG metrics from supplier reports
- **Metrics Analyzed**:
  - Environmental indicators
  - Social performance
  - Governance structures
- **Output**: ESG scores and recommendations

## 📊 Data Files

### `emission_factors.json`
Contains CO2 emission factors for different transportation modes and energy sources.

**Structure**:
```json
{
  "transport_modes": {
    "air": 0.255,
    "truck": 0.120,
    "rail": 0.041,
    "sea": 0.011
  },
  "energy_sources": {
    "coal": 0.820,
    "natural_gas": 0.490,
    "renewable": 0.0
  }
}
```

### `scenarios.json`
Predefined supply chain scenarios with optimization parameters.

**Example**:
```json
{
  "scenarios": [
    {
      "name": "Green Fleet",
      "modal_shift_pct": 50,
      "renewable_increase_pct": 30
    }
  ]
}
```

### `sample_*.csv` Files
Sample datasets for testing and development:
- `sample_logistics_routes.csv` - Shipping route data
- `sample_operational_logs.csv` - Operational metrics
- `sample_supplier_esg.csv` - Supplier ESG information

## 👨‍💻 Development

### Project Setup for Development

1. **Install development dependencies**
   ```bash
   pip install -r requirement.txt
   ```

2. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run in development mode**
   ```bash
   FLASK_ENV=development FLASK_DEBUG=1 python app.py
   ```

### Code Structure

- **`app.py`**: Main Flask application with all route definitions
- **`models/`**: Machine learning model implementations
- **`utils/`**: Helper functions for data processing and validation
- **`data/`**: Static data files and resources

### Key Components in app.py

1. **Custom JSON Encoder**: Handles NaN/Inf values in JSON responses
2. **Model Initialization**: Lazy loading of ML models
3. **Error Handling**: Comprehensive error responses
4. **Data Validation**: Input validation for all endpoints
5. **Report Management**: Asynchronous report generation tracking

### Adding New Features

1. Create model class in `models/`
2. Add corresponding API endpoint in `app.py`
3. Add input validation in `utils/validators.py`
4. Update documentation

## 🐳 Docker Configuration

### Dockerfile
- Base image: Python 3.10-slim
- Build optimizations for reduced size
- System dependencies: gcc, g++ for compilation
- Working directory: `/app`
- Port exposure: 5000

### Docker Compose Services
- Service name: `carbon-backend`
- Container name: `carbon-backend`
- Volume mounts for data persistence
- Environment variables for model optimization

### Environment Variables
```
FLASK_ENV=development
PYTHONUNBUFFERED=1
MINDSPORE_DEVICE_TARGET=CPU
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
```

## 📈 Performance Considerations

- **Model Inference**: Optimized for CPU execution via MindSpore
- **Batch Processing**: Support for bulk operations
- **Data Caching**: Efficient data loading and preprocessing
- **Parallel Processing**: Multi-threaded report generation
- **Memory Management**: Proper tensor cleanup and optimization

## 🔒 Security Notes

- CORS enabled for development (adjust for production)
- Input validation on all endpoints
- Error messages sanitized to prevent information leakage
- File uploads restricted to designated directories

## 📝 License

This project is part of the SSCA (Supply Chain Carbon Analytics) initiative.

## 📧 Support

For issues, questions, or contributions, please contact the development team.

---

**Last Updated**: December 2025  
**Version**: 1.0.0

