# Real-time House Price Prediction System - Redpanda Hackathon Project

A real-time prediction system built for the Redpanda Hackathon, showcasing the integration of Redpanda Connect and Redpanda Data Transform features. The system demonstrates an innovative AI solution that predicts house prices using a machine learning model with real-time data streaming capabilities.

## Hackathon Requirements Met

This project implements two key Redpanda technologies:

1. **Redpanda Connect**: 
   - Implements seamless connection between the Kafka-compatible broker and our prediction service
   - Manages data flow between different components of the system
   - Ensures reliable message delivery and processing

2. **Redpanda Data Transform**:
   - Transforms data between `prediction_topic` and `prediction_result_topic`
   - Processes raw input data for model prediction
   - Formats prediction results for frontend consumption

## Project Overview

This project demonstrates real-time AI/ML capabilities by:
- Processing streaming data for house price predictions
- Implementing real-time updates using Server-Sent Events
- Showcasing scalable architecture using Redpanda's features

## Architecture Overview

```
[Frontend (HTML/SSE)] <--> [Flask Server]
         ^                      ^
         |                      |
         v                      v
[Redpanda Connect] <--> [Redpanda Data Transform]
         ^                      ^
         |                      |
         v                      v
[prediction_topic] <--> [prediction_result_topic]
         ^                      ^
         |                      |
         v                      v
    [ML Model] <----------> [Redis Cache]
```

## Technologies Used

- **Core Requirements**:
  - Redpanda Connect
  - Redpanda Data Transform
  - Machine Learning Model (scikit-learn)

- **Supporting Technologies**:
  - Flask: Web framework
  - Redis: Event streaming support
  - Docker & Docker Compose: Containerization
  - NumPy/Pandas: Data processing
  - APScheduler: Background processing

## Quick Start

1. Clone the repository
2. Start the services:
```bash
docker-compose up -d
```

The system starts:
- Redpanda broker (port 9092)
- Redis (port 6379)
- Model service (port 5001)

## How It Works

1. **Data Input Flow**:
   - Frontend submits prediction request
   - Redpanda Connect routes request to `prediction_topic`
   - Data Transform processes input for model consumption

2. **Prediction Process**:
   - Background worker processes prediction request
   - Results published to `prediction_result_topic`
   - Data Transform formats results for frontend

3. **Real-time Updates**:
   - SSE provides live updates to frontend
   - Redis manages event streaming
   - Frontend updates dynamically

## API Documentation

### Endpoints
- `GET /`: Home page with prediction interface
- `POST /predict_api`: Submit prediction request
  - Input: JSON with house features
  - Output: Prediction status
- `GET /stream`: SSE endpoint for real-time updates

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Redpanda Account for Connect access

### Local Development
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export REDIS_URL=redis://localhost:6380
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Run the application
python app.py
```

## Demo

Access the live demo at: [Here](https://housevalue.lawrencesegun.xyz/)

## Repository Structure

```
├── app.py              # Main Flask application
├── producer.py         # Kafka producer implementation
├── Dockerfile         # Container configuration
├── docker-compose.yml # Service orchestration
├── templates/         # Frontend templates
├── regmodel.pkl      # Trained ML model
└── scaling.pkl       # Data scaler
```

## Future Enhancements

1. **Scalability**:
   - Implement multiple Redpanda nodes
   - Add load balancing
   - Scale prediction workers

2. **Features**:
   - Add more ML models
   - Implement A/B testing
   - Add model performance monitoring

3. **UI/UX**:
   - Enhanced visualization
   - Interactive data input
   - Historical prediction tracking