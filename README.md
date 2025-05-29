# Stock Sentiment Analysis Project

## Project Overview

This project analyzes financial news sentiment and its correlation with stock market movements for Nova Financial Solutions. The analysis combines natural language processing with financial data analysis to derive actionable trading insights.

## Project Structure

```
├── .vscode/              # VSCode configuration
├── .github/              # GitHub Actions workflows
├── data/                 # Data directory
│   ├── raw/             # Raw data files
│   └── processed/       # Processed data files
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code
├── tests/              # Unit tests
├── scripts/            # Utility scripts
└── requirements.txt    # Project dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Environment Setup

1. Clone the repository:

```bash
git clone [repository-url]
cd stock-sentiment-analysis
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Data

The project uses financial news data with the following structure:

- headline: News article headline
- url: Article URL
- publisher: Article publisher
- date: Publication date (UTC-4)
- stock: Stock ticker symbol

## Development Guidelines

- Use feature branches for development
- Follow PEP 8 style guide
- Write unit tests for new features
- Update requirements.txt when adding dependencies

## Testing

Run tests using:

```bash
python -m pytest tests/
```

## Contributing

1. Create a feature branch
2. Make changes
3. Run tests
4. Submit pull request

## License

[License information]
