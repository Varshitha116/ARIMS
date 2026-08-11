# ARIMS (Agentic AI-Based Autonomous Road Infrastructure Maintenance System)

## Project Overview

ARIMS is an AI-powered autonomous system designed to maintain road infrastructure integrity through machine learning and computer vision. It detects road defects, predicts degradation, and coordinates maintenance activities using a multi-agent architecture.

## Key Features

- ✅ Real-time defect detection (cracks, potholes, surface failures) via computer vision
- ✅ Degradation prediction using historical data and environmental factors
- ✅ Multi-agent scheduling for repair task prioritization
- ✅ Interactive dashboard with geospatial visualizations
- ✅ RESTful API for system integration
- ✅ Production-ready monitoring and logging

## Requirements

- Python 3.11+
- Streamlit
- OpenCV
- Pillow
- scikit-learn
- PostgreSQL with PostGIS extension

## Installation

```bash
# Clone repository
cd road-ai-project
# Install dependencies
pip install -r requirements.txt
# Start dashboard
streamlit run main.py
# (For containerized deployment: docker-compose up)
```

## Usage

1. **Dashboard Access**: Visit http://localhost:8501 in browser after running `streamlit run main.py`
2. **API Usage**:
   - POST `/detect` with image data for defect analysis
   - GET `/schedule` to view current repair plans
   - POST `/schedule` to submit repair requests
3. **Image Input**: Accepts JPEG/PNG files via dashboard or API

## Documentation

- [Software Design Document (SDD)](/SDD.md)
- [Project Journal](/PROJECT_JOURNAL.md)
- [RAML API Specifications](/docs/api-spec.raml)

## Contributing

Contributions welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes with descriptive messages
4. Submit pull request with background context

For major changes, discuss first in PROJECT_JOURNAL.md

## License

MIT License

## Roadmap

- [ ] Multi-modal input integration (SAR imagery, sensor data)
- [ ] Reinforcement learning scheduler
- [ ] Edge deployment optimization

## Visual Branding

![ARIMS Logo](https://example.com/arims-logo.png)

*— Auto-generated README v1.0*