# Solar+Storage Arbitrage Value Calculator

A comprehensive tool that demonstrates how battery storage unlocks economic value for utility-scale solar assets through energy arbitrage.

## Overview

This application quantifies the economic benefits of adding battery storage to solar projects by:
- Storing cheap midday solar energy
- Discharging during expensive evening peaks
- Calculating revenue uplift and payback periods
- Visualizing arbitrage strategies

## Features

### Core Functionality
- **Real-time Price Data**: Synthetic CAISO and ERCOT price data with realistic market characteristics
- **Solar Generation Modeling**: Hourly solar profiles using sine wave functions
- **Storage Dispatch Algorithm**: Intelligent charge/discharge decisions based on daily average prices
- **Financial Analysis**: Revenue uplift, payback periods, and ROI calculations

### Interactive Dashboard
- **System Configuration**: Adjustable solar capacity, battery power, and duration
- **Financial Parameters**: Battery CAPEX, efficiency, and market selection
- **Real-time Visualization**: 24-hour dispatch profiles and battery state of charge
- **Export Capabilities**: Download results as CSV files

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run arbitrage_dashboard.py
```

## Usage

### Basic Workflow
1. **Select Market**: Choose between California (CAISO) or Texas (ERCOT) price data
2. **Configure System**: Set solar capacity, battery power, and duration
3. **Set Financials**: Input battery CAPEX and efficiency assumptions
4. **Run Simulation**: Click "Run Simulation" to calculate results
5. **Analyze Results**: Review KPIs, visualizations, and download data

### Key Parameters
- **Solar System Size**: 10-500 MW (default: 100 MW)
- **Battery Power**: 5-250 MW (default: 50 MW)
- **Battery Duration**: 2-6 hours (default: 4 hours)
- **Battery CAPEX**: $200-600/kWh (default: $350/kWh)
- **Round-Trip Efficiency**: 80-95% (default: 88%)

## Technical Architecture

### Core Modules

#### `arbitrage_model.py`
- **SolarStorageArbitrage**: Main model class
- **Data Handling**: Price data loading and preprocessing
- **Solar Modeling**: Synthetic generation profiles
- **Dispatch Algorithm**: Storage optimization logic
- **Financial Calculations**: Revenue and payback analysis

#### `arbitrage_dashboard.py`
- **Streamlit Interface**: Interactive web dashboard
- **Control Panel**: System configuration and parameters
- **Visualizations**: Charts and metrics display
- **Export Functions**: Data download capabilities

### Algorithm Details

#### Storage Dispatch Logic
1. **Calculate Daily Average Price**: Baseline for charge/discharge decisions
2. **Charge Signal**: When market price < daily average
   - Store available solar energy
   - Apply round-trip efficiency losses
3. **Discharge Signal**: When market price > daily average
   - Discharge stored energy to grid
   - Apply efficiency losses on discharge

#### Financial Metrics
- **Revenue Uplift**: Additional annual revenue from storage
- **Revenue Increase %**: Percentage improvement over standalone solar
- **Payback Period**: Years to recover battery investment
- **Battery Cost**: Total capital investment in storage

## Market Data

### Synthetic Price Data
- **CAISO (California)**: Duck curve characteristics with midday lows and evening peaks
- **ERCOT (Texas)**: High volatility with extreme price events
- **Realistic Patterns**: Seasonal variation, daily cycles, and price spikes

### Solar Generation
- **Sine Wave Model**: Peak at noon, zero at night
- **Normalized Profiles**: Consistent with utility-scale solar characteristics
- **Hourly Resolution**: 8,760 hours per year

## Visualization Features

### Main Arbitrage Chart
- **Price Line**: Electricity prices over time
- **Energy Bars**: Solar generation vs battery discharge
- **Battery SOC**: State of charge percentage
- **Interactive Controls**: Date selection and time range

### Key Performance Indicators
- **Annual Revenue Uplift**: Dollar value of storage addition
- **Revenue Increase %**: Relative improvement metric
- **Payback Period**: Investment recovery timeline

## Business Value

This tool demonstrates:
- **Quantitative Analysis**: Data-driven storage valuation
- **Market Understanding**: Price arbitrage opportunities
- **Financial Modeling**: Investment decision support
- **Visual Communication**: Clear presentation of complex concepts

## Future Enhancements

### Data Integration
- **EIA API**: Real-time price data from U.S. Energy Information Administration
- **CAISO API**: Live California market data
- **ERCOT API**: Texas market integration

### Advanced Features
- **Multiple Markets**: Additional electricity markets
- **Weather Integration**: Solar generation based on weather data
- **Degradation Modeling**: Battery and solar panel degradation
- **Grid Services**: Ancillary service revenue streams

## Technical Requirements

- **Python 3.8+**
- **Streamlit 1.28+**
- **Pandas 2.0+**
- **NumPy 1.24+**
- **Plotly 5.15+**

## License

This project is designed for educational and demonstration purposes in the renewable energy sector.
