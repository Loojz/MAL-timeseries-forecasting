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
| Marvin  | @izproxy-cl     | Gold (XAU/USD)   |
| Alper   | @alperbildiren  | EUR/USD          |
| Luis    | @Loojz          | Crude Oil (WTI)  |

These series are economically linked: oil is priced in USD, gold reacts
to USD strength, and EUR/USD reflects monetary policy. This makes them
ideal for multivariate modeling in Part 3.

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

## Contributing

See CONTRIBUTING.md for branching strategy, commit conventions,
and code review process.

## Milestones

- 04.05.2026: Repository setup and topic selection
- 11.05.2026: Univariate analyses complete
- 18.05.2026: Multivariate pipeline and final presentation

## License

MIT License - see LICENSE file.

