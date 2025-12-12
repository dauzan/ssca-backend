# SSCA Backend — Sustainable Supply Chain AI (SSCA) Advisor

This repository contains the Flask backend for the SSCA Advisor (Sustainable Supply Chain AI): a set of services and machine-learning components used to forecast emissions, score supplier sustainability risk, analyze regulations using NLP, and optimize supply chain scenarios for cost and carbon.

Contents in this updated README:
- Overview
- Quick start (Docker + local)
- API summary with main endpoints
- Project layout and data
- Models & how to run them
- Troubleshooting & next steps

---

## Overview

SSCA Backend exposes REST endpoints for model inference and orchestration of analytics tasks. It is intended to be run locally for development or inside Docker for deployment.

Primary capabilities
- Emission forecasting (time-series models)
- Supplier ESG & risk scoring
- Regulatory text analysis (NER / extraction)
- Multi-objective supply-chain optimization
- Report generation and download

## Quick Start

Recommended: use Docker Compose to build and run all services.

Using Docker (recommended)
```bash
cd backend
docker-compose up --build
```
The API will be available at http://localhost:5000 (default).

Local Python (dev)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
python app.py
```

Health check
>>>>>>> cb58bff (release(change): change the docker file into more production mode and clean up the README.md file)
```bash
curl http://localhost:5000/health
```

<<<<<<< HEAD
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
=======
---

## API Summary (main endpoints)

Base URL: http://localhost:5000/api/v1

- POST /forecast/emissions — run emission forecasting (input: historical operational data)
- POST /score/supplier — compute supplier risk score (input: supplier ESG fields)
- POST /analyze/regulation — extract entities from a regulation text
- POST /batch/analyze/regulations — batch regulatory document processing
- POST /optimize/supply-chain — run scenario optimization
- POST /generate/report — request report generation
- GET /report/status/<report_id> — check report progress
- GET /download/<report_id> — download generated report
- GET /suppliers — list suppliers
- POST /suppliers/upload — bulk supplier upload (CSV)

For request/response contracts, inspect the route handlers in `app.py` and the helper functions in `utils/`.

---

## Project Layout

backend/
- app.py                # Flask app and route bindings
- Dockerfile
- docker-compose.yaml
- requirement.txt       # Python deps
- README.md
- data/                 # sample data and configuration (emission factors, scenarios)
- models/               # model implementations and wrappers
- utils/                # data processing and validation helpers
- save_models/          # persisted model artifacts

Files you will commonly change
- `models/*.py` — model inference and wrappers
- `utils/data_processor.py` — transforms and validators for incoming payloads
- `app.py` — route wiring and simple orchestration

---

## Data & Configuration

- `data/emission_factors.json` — factors per transport and energy source
- `data/sample_*` — example CSVs used in tests and docs
- `data/scenarios.json` — pre-configured scenarios for the optimizer

Inspect `data/` before running models locally; missing files will cause predictable errors.

---

## Models (brief)

- `models/emission_forecasting.py`: time-series model (LSTM/attention style) trained on 30-day windows. Input: historical operational metrics; Output: CO2e forecast.
- `models/supplier_risk.py`: scoring routine combining ESG features into a composite risk score.
- `models/regulatory_nlp.py`: transformer/NER utilities for extracting thresholds, deadlines, penalties, scopes.
- `models/optimization.py`: scenario optimizer (multi-objective) for modal shifts and renewable integration.

Model artifacts are expected under `save_models/` — if you have pre-trained artifacts, place them there and adjust paths in the model loader functions.

---

## Development notes

- Use the provided `requirement.txt` to install dependencies.
- Keep the virtual environment isolated for local development.
- Endpoint implementations are intentionally small — extend validation in `utils/validators.py` before exposing models publicly.

Testing and quick checks
```bash
# run a quick smoke check
curl -X GET http://localhost:5000/health

# POST a small payload to test forecasting (example payload shape in comments in app.py)
```

---

## Docker

The repository includes a `Dockerfile` and `docker-compose.yaml` to streamline running the service inside containers. Use `docker-compose up --build` to recreate images after code changes.

---

## Production (Gunicorn)

To run the Flask app in production, we recommend using `gunicorn` behind a reverse proxy (e.g., Nginx).

1. Install `gunicorn` in your production environment or virtualenv:

```bash
pip install gunicorn
```

2. Recommended command (adjust `--workers` based on CPU):

```bash
gunicorn --workers 3 --worker-class gthread --threads 4 --bind 0.0.0.0:8000 app:app --access-logfile - --error-logfile - --log-level info
```

- Worker sizing suggestion: `workers = (2 x $CPU) + 1`.
- Use `--bind 0.0.0.0:8000` (or 5000) depending on your proxy and firewall rules.
- Ensure `app:app` matches the Flask application object name and import path in `app.py`.

3. Example `systemd` unit (optional — update paths and user):

```ini
[Unit]
Description=SSCA Advisor Gunicorn
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/backend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 app:app

[Install]
WantedBy=multi-user.target
```

4. Reverse proxy (recommended):
- Put Nginx in front of Gunicorn to handle TLS, static files, buffering and client timeouts.

Notes:
- Set `FLASK_ENV=production` and other environment variables securely (do not keep secrets in source).
- Monitor logs and tune worker class (`sync`, `gthread`, `gevent`) and counts according to load and IO characteristics.


## Troubleshooting

- If startup fails, check logs from Docker Compose or `python app.py` output.
- Common issues: missing data files, dependency mismatches, or model artifacts not present.
- If a model import fails, verify the Python path and ensure the package versions in `requirement.txt` match your environment.

---

## Next steps I can help with

- Add example request/response JSON for each endpoint
- Create a minimal Postman collection or OpenAPI spec
- Add unit tests for `utils/` and core endpoints

If you'd like, I can (a) add example payloads and responses to this README, (b) generate an OpenAPI spec from `app.py`, or (c) run the service and exercise a health-check endpoint.

---

File: [backend/README.md](backend/README.md)

>>>>>>> cb58bff (release(change): change the docker file into more production mode and clean up the README.md file)

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
<<<<<<< HEAD
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

=======
- Service name: `ouddockdock/huawei-mindspore-backend`
- Container name: `ouddockdock/huawei-mindspore-backend`
- Volume mounts for data persistence
- Environment variables for model optimization

>>>>>>> cb58bff (release(change): change the docker file into more production mode and clean up the README.md file)
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

