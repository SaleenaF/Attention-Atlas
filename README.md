# Attention Atlas: Creator Intelligence Framework

[![Live Dashboard](https://img.shields.io/badge/Streamlit-Live_Dashboard-red?logo=streamlit)](https://attention-atlas-analytics.streamlit.app/)

An advanced analytics engineering and predictive framework built to ingest, transform, and model multi-dimensional audience retention and performance matrices from creator studio datasets.

An advanced analytics engineering and predictive framework built to ingest, transform, and model multi-dimensional audience retention and performance matrices from creator studio datasets.

## Project Architecture & Workflow

1. **ETL Pipeline (`src/pipeline.py`)**: Ingests multi-tab source data, applies dynamic format classification thresholds based on video properties, filters out aggregate total anomalies, and executes an isolated data anonymization routine.
2. **Data Warehouse (`attention_atlas.db`)**: A streamlined SQLite database optimized for minimal storage footprint, featuring individual views for categorical video performance, historical channel timelines, and geodemographic vectors.
3. **Machine Learning Backend (`src/model.py`)**: Trains an audience engagement predictor utilizing advanced feature weighting, saving the serialized artifact (`view_predictor.joblib`) for real-time inference.
4. **Interactive Intelligence Console (`dashboard.py`)**: A production-ready Streamlit application equipped with Plotly interactive visualization modules, performance distribution matrices, and data-driven operational recommendations.

---

## Data Privacy & Compliance Note

To maintain brand anonymity while presenting a production-grade creator intelligence framework, all video identifiers (`Content ID`) and character asset metadata strings (`Video Title`) are fully masked and anonymized utilizing an isolated tokenization sequence during the ETL ingestion pipeline. 

All structural analytics integrity—including historical retention ratios, timeseries data matrices, content format distributions, geographic audience splits, and predictive ML feature weights—remains completely authentic.

---

## Installation & Deployment

### 1. Environment Setup
Clone this repository and ensure Python 3.10+ is installed locally. Populate your virtual environment with required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
