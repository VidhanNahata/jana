# Civic Complaint Forensics Pipeline

An end-to-end AI-powered system for validating civic complaint authenticity and automating priority-based routing.

## Architecture Overview

```
Citizen Upload → ELA CNN Verification → IndicSBERT Deduplication → RF/ITIL Priority → Department Dispatch
                      ↓                         ↓                        ↓
               Forgery Rejected          Master Ticket Merge       SLA Enforcement
```

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Bootstrap data directories and seed datasets
python bootstrap_project.py

# 3. Run standalone scripts (no external services needed)
python itil_priority_matrix.py
python master_ticket_system.py
python train_priority_rf.py
python ml_evaluate_baselines.py
python simulate_bengaluru.py
```

## Project Structure

```
civic_dataset_tools/
├── bootstrap_project.py          # One-shot project setup (run this first!)
├── requirements.txt              # Python dependencies
│
├── ── Phase 1: Data Collection & Preprocessing ──
├── setup_dataset.py              # Create authentic/ and tampered/ folders
├── tamper_tool.py                # Generate copy-move and splicing forgeries
├── ela_pipeline.py               # Batch ELA preprocessing
├── exif_pipeline.py              # EXIF metadata extraction & anomaly flagging
├── generate_synthetic_pairs.py   # Multilingual complaint pair generation
├── indicbert_preprocess.py       # Unicode normalization & SentencePiece tokenization
├── sts_dataset_generator.py      # Cross-lingual STS training data
│
├── ── Phase 2: Model Training & AI Modules ──
├── ela_dataset.py                # PyTorch Dataset with on-the-fly ELA
├── train_forgery_cnn.py          # ResNet-50 binary forgery classifier
├── blip2_api.py                  # BLIP-2 zero-shot visual description API
├── train_indicbert.py            # IndicBERT 4-class complaint classifier
├── indicsbert_deduplication.py   # Semantic deduplication with geospatial bounds
├── master_ticket_system.py       # SQLAlchemy ORM for Master/Child tickets
├── train_priority_rf.py          # Random Forest priority scorer
├── itil_priority_matrix.py       # ITIL 3×3 priority matrix + RF override
│
├── ── Phase 3: Backend Infrastructure ──
├── backend_app.py                # FastAPI server with PostGIS integration
│
├── ── Phase 4: Frontend ──
├── frontend/
│   ├── src/components/
│   │   ├── CitizenPortal.jsx     # Complaint submission + GPS + tracking
│   │   └── AdminDashboard.jsx    # GIS heatmaps + HITL confidence review
│   ├── src/App.jsx               # React Router setup
│   ├── src/main.jsx              # Vite entry point
│   └── package.json              # Frontend dependencies
│
├── ── Phase 5: Evaluation & Docs ──
├── ml_evaluate_baselines.py      # Benchmark CNN/NLP/RF against baselines
└── simulate_bengaluru.py         # End-to-end Bengaluru integration test
```

## ML Training (GPU recommended)

```bash
# Image forgery classifier (requires ela_output/ from bootstrap)
python train_forgery_cnn.py

# IndicBERT text classifier (requires complaints_text.csv from bootstrap)
python train_indicbert.py

# Random Forest (generates its own synthetic data)
python train_priority_rf.py
```

## Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Backend API

```bash
# Requires PostgreSQL + PostGIS for full functionality
# Falls back to SQLite for basic testing
python backend_app.py
# Swagger UI at http://localhost:8000/docs
```

## Key API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/complaints/ingest` | Submit complaint through AI pipeline |
| GET | `/api/v1/gis/hotspots` | Get spatial clusters (100m DBSCAN) |
| GET | `/api/v1/departments/{name}/tasks` | Department-specific task queue |
| POST | `/analyze-image` | BLIP-2 visual description (blip2_api.py) |

## SLA Priority Matrix

| Priority | Response Time | Criteria |
|----------|--------------|----------|
| P1 Critical | Immediate (≤1hr) | High Severity + High Urgency |
| P2 High | 24 hours | High + Medium combination |
| P3 Medium | 72 hours | Medium Severity |
| P4 Low | 1 week | Isolated low-severity incidents |

## Tech Stack

- **Image Forensics**: ELA + ResNet-50 CNN (PyTorch)
- **Visual AI**: BLIP-2 (HuggingFace Transformers)
- **NLP**: IndicBERT, IndicSBERT (L3Cube), LaBSE
- **Priority ML**: Random Forest + ITIL Matrix
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL/PostGIS
- **Frontend**: React + Bootstrap + React Router
- **GIS**: PostGIS ST_ClusterDBSCAN, SRID 4326/3395
