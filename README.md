# Solar + Storage Arbitrage Calculator

A comprehensive Streamlit dashboard for analyzing the economic value of adding battery storage to solar farms through energy arbitrage optimization.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📊 Features

### Real-World Data Analysis
- **EIA API Integration**: Fetches real-time electricity price data from CAISO and ERCOT markets
- **Historical Price Patterns**: Analyzes seasonal variations, duck curve effects, and market volatility
- **Optimized Arbitrage**: Simulates intelligent battery charging/discharging based on price thresholds

### Synthetic Data Modeling
- **Market Simulation**: Generates realistic price patterns with seasonal variation
- **Duck Curve Modeling**: Replicates California's characteristic demand patterns
- **Volatility Analysis**: Includes price spikes and market dynamics

### Key Metrics
- **Revenue Uplift**: Additional revenue from battery arbitrage
- **Battery Cycles**: Operational efficiency metrics
- **Solar-Only vs Solar+Storage**: Comparative revenue analysis
- **Interactive Visualizations**: Real-time dispatch charts and price patterns

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/solar-storage-arbitrage.git
   cd solar-storage-arbitrage
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   # For real-world data (requires EIA API key)
   streamlit run arb_api_dash.py
   
   # For synthetic data (no API key required)
   streamlit run arb_synth_dash.py
   ```

## 📁 Project Structure

```
solar-storage-arbitrage/
├── arb_api_dash.py          # Real-world data dashboard
├── arb_synth_dash.py        # Synthetic data dashboard
├── arbit.py                 # Core arbitrage model
├── arbitrage_api.py         # EIA API integration
├── aribtrage_synth.py       # Synthetic data generation
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## 🔧 Configuration

### EIA API Setup (for real-world data)
1. Get a free API key from [EIA.gov](https://www.eia.gov/opendata/)
2. The API key is hardcoded in the application for demo purposes
3. For production use, set as environment variable: `EIA_API_KEY`

### System Parameters
- **Solar System Size**: 50-500 MW
- **Battery Power**: 25-250 MW  
- **Battery Duration**: 2-6 hours
- **Round-trip Efficiency**: 80-95%
- **Battery CAPEX**: $100-250/kWh

## 📈 Usage

### Real-World Data Mode
1. Select market (CAISO or ERCOT)
2. Configure system parameters
3. Click "Run Simulation"
4. View results and interactive charts

### Synthetic Data Mode
1. Configure system parameters
2. Click "Run Simulation" 
3. Analyze generated market patterns
4. Compare different scenarios

## 🧮 Algorithm

The arbitrage optimization uses a dynamic threshold strategy:

1. **Price Analysis**: Calculates daily price quantiles (33rd and 67th percentiles)
2. **Charging Logic**: Charges battery when prices are in bottom 33%
3. **Discharging Logic**: Discharges battery when prices are in top 33%
4. **Solar Priority**: Uses solar generation first, then grid power if profitable
5. **Efficiency Modeling**: Accounts for round-trip losses and power constraints

## 📊 Key Metrics Explained

- **Revenue Uplift**: Additional annual revenue from battery arbitrage
- **Revenue Increase %**: Percentage improvement over solar-only revenue
- **Battery Cycles**: Total charge/discharge cycles per year
- **Solar-Only Revenue**: Revenue without battery storage
- **Solar+Storage Revenue**: Total revenue with optimized arbitrage

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the appropriate dashboard file
   - Deploy!

### Environment Variables
For production deployment, set:
- `EIA_API_KEY`: Your EIA API key (for real-world data)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EIA (Energy Information Administration)** for providing electricity price data
- **CAISO and ERCOT** for market data access
- **Streamlit** for the amazing dashboard framework
- **Plotly** for interactive visualizations

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for the clean energy transition**
