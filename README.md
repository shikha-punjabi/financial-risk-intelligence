# AI-Powered Financial News Risk Intelligence Platform

This project provides a modular, Python-based platform for analyzing financial news and assessing risk using data science, machine learning, and NLP techniques.

## Project Structure
- `src/` - Source code (data, models, agents, dashboard, utils)
- `data/` - Data storage (raw, processed, models)
- `config/` - Configuration files
- `tests/` - Unit and integration tests

## Setup
1. Create a virtual environment:
   ```sh
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - Windows: `./.venv/Scripts/activate`
   - macOS/Linux: `source .venv/bin/activate`
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` as needed.

## Configuration

Risk thresholds and sector mappings are now in `config/settings.py`:

```python
RISK_THRESHOLDS = {
    "sentiment_critical": -0.7,
    "risk_article_threshold": 0.3,
    ...
}
SECTOR_KEYWORDS = {
    "Technology": [...],
    ...
}
```

You can update these to tune the platform’s behavior.

## Usage
- Extend the platform by adding new agents, models, or dashboards in the respective folders.
- Use Streamlit for interactive dashboards.

## License
This project uses only free and open-source libraries.
