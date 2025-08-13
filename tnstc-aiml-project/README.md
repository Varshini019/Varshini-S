# TNSTC AI/ML Student Project: Bus Demand Forecasting & Schedule Helper

This student project predicts per-stop passenger demand for TNSTC bus services and exposes a simple API to help planners estimate crowding and adjust schedules.

## What you get
- Synthetic TNSTC-like dataset generator
- Training script (scikit-learn pipeline)
- Minimal FastAPI prediction service
- Clear steps to run end-to-end locally

## Folder structure
- `src/`: code for data generation, training, and API
- `data/`: generated CSVs
- `models/`: trained model artifacts

## Quickstart
1) Create and activate virtual environment
```bash
cd /workspace/tnstc-aiml-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Generate synthetic data
```bash
python -m src.generate_synthetic_data --num_routes 12 --stops_per_route 15 --num_days 90 --start_date 2025-01-01 --seed 42
```

3) Train the model
```bash
python -m src.train_demand_forecast --test_days 14 --random_state 42
```

4) Serve the API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

5) Example request
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "route_id": "R03",
    "stop_id": "R03_S07",
    "timestamp": "2025-08-15T08:30:00",
    "is_holiday": false,
    "weather_index": 0.6
  }'
```

## Ideas to extend
- Add real GTFS-like schedules and GPS traces (if available)
- Incorporate weather and event data for accuracy
- Build a small dashboard (Streamlit) for planners
- Add anomaly detection for schedule adherence and bunching
- Ethical considerations: privacy, fairness across routes/regions

## License
For educational use.