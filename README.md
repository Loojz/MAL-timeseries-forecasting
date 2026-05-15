# MAL Time Series Forecasting

University project at THWS (Technische Hochschule Wuerzburg-Schweinfurt)
Course: Vertiefung Business Analytics (Prof. Dr. Christian Menden)

## Project Overview

This project explores time series forecasting using both classical
statistical methods (Box-Jenkins / ARIMA) and modern machine learning
approaches. We analyze three macroeconomically interconnected assets.

The work is structured in three parts:

1. Part 1: Git Repository - Professional collaboration practices
2. Part 2: Univariate ARIMA analysis per team member
3. Part 3: Multivariate forecasting pipeline

## Team and Time Series

| Member  | GitHub          | Time Series      |
|---------|-----------------|------------------|
| Marvin  | @izproxy-cl     | BTC (BTC/USD)    |
| Alper   | @alperbildiren  | EUR/USD          |
| Luis    | @Loojz          | Gold (XAU/USD)   |

These series mix traditional macro assets (gold, EUR/USD) with a digital asset (BTC), 
allowing us to study both classical economic linkages and how cryptocurrencies behave 
relative to fiat and commodity markets — a particularly interesting question for the 
multivariate analysis in Part 3.

## Repository Structure

- data/raw          Original downloads
- data/processed    Cleaned data
- notebooks/        Jupyter notebooks
- src/              Python modules
- reports/          Presentations and figures

## Setup

    git clone git@github.com:Loojz/MAL-timeseries-forecasting.git
    cd MAL-timeseries-forecasting
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

### TimeGPT-2.1 (optional)

To enable TimeGPT-2.1 forecasts in the Research tab:

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your API key:
   ```
   NIXTLA_API_KEY=your_key_here
   ```

3. Restart the app — TimeGPT-2.1 activates automatically:
   ```bash
   streamlit run app.py
   ```

Get a free API key at [dashboard.nixtla.io](https://dashboard.nixtla.io).
Available models: `timegpt-2.1` (recommended), `timegpt-2-pro` (highest accuracy), `timegpt-2-mini` (fastest).

## Contributing

See CONTRIBUTING.md for branching strategy, commit conventions,
and code review process.

## Milestones

- 04.05.2026: Repository setup and topic selection
- 11.05.2026: Univariate analyses complete
- 18.05.2026: Multivariate pipeline and final presentation

## License

MIT License - see LICENSE file.

