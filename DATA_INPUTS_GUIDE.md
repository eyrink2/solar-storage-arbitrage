# Solar+Storage Arbitrage Value Calculator - Data Inputs Guide

## Overview
This document explains exactly what data inputs are required to run the Solar+Storage Arbitrage Value Calculator and how the system processes them.

## Required Data Inputs

### 1. Price Market Selection
**Input Type:** Dropdown selection
**Options:** 
- California (CAISO) - Duck curve characteristics, moderate volatility
- Texas (ERCOT) - High volatility, extreme price events

**What it does:** Determines which synthetic electricity price data to generate
**Default:** California (CAISO)

### 2. Solar System Size
**Input Type:** Slider
**Range:** 10-500 MW
**Default:** 100 MW
**Step:** 10 MW increments

**What it does:** Sets the peak solar capacity in megawatts
**Impact:** Larger systems generate more energy but require more storage

### 3. Battery Power
**Input Type:** Slider  
**Range:** 5-250 MW
**Default:** 50 MW
**Step:** 5 MW increments

**What it does:** Sets the maximum charge/discharge rate in megawatts
**Impact:** Higher power allows faster charging/discharging

### 4. Battery Duration
**Input Type:** Slider
**Range:** 2-6 hours
**Default:** 4 hours
**Step:** 1 hour increments

**What it does:** Sets how long the battery can discharge at full power
**Calculated:** Battery Capacity (MWh) = Power (MW) × Duration (hours)

### 5. Battery CAPEX
**Input Type:** Slider
**Range:** $200-600/kWh
**Default:** $350/kWh
**Step:** $25/kWh increments

**What it does:** Sets the capital cost per kWh of battery capacity
**Impact:** Higher costs increase payback period

### 6. Round-Trip Efficiency
**Input Type:** Slider
**Range:** 80-95%
**Default:** 88%
**Step:** 1% increments

**What it does:** Sets the overall efficiency of charging and discharging
**Impact:** Lower efficiency reduces arbitrage value

## Data Processing Flow

### Step 1: Price Data Generation
```
Input: Market Selection (CAISO/ERCOT)
Output: 8,760 hours of synthetic price data
Process:
- Creates realistic seasonal patterns
- Adds daily duck curve (low midday, high evening)
- Includes market volatility and extreme events
- Ensures positive prices
```

### Step 2: Solar Profile Generation
```
Input: Solar System Size (MW)
Output: 8,760 hours of solar generation (kW)
Process:
- Uses sine wave function peaking at noon
- Zero generation at night (6 PM - 6 AM)
- Scales to system size
```

### Step 3: Battery System Setup
```
Input: Power (MW), Duration (hours), CAPEX ($/kWh), Efficiency (%)
Calculations:
- Capacity (MWh) = Power × Duration
- Total Cost = Capacity × CAPEX
- Efficiency factor = sqrt(efficiency)
```

### Step 4: Arbitrage Simulation
```
Process: For each hour (8,760 iterations)
1. Get current price and solar generation
2. Calculate daily average price (dispatch threshold)
3. If price < daily average: CHARGE battery
4. If price > daily average: DISCHARGE battery
5. Calculate revenues for both scenarios
6. Track battery state of charge
```

## Key Calculations

### Revenue Calculations
- **Standalone Solar Revenue** = Solar Generation × Market Price
- **Hybrid Revenue** = (Solar + Battery Discharge) × Market Price
- **Revenue Uplift** = Hybrid Revenue - Standalone Revenue

### Financial Metrics
- **Revenue Increase %** = (Uplift / Standalone Revenue) × 100
- **Battery Cost** = Capacity (kWh) × CAPEX ($/kWh)
- **Payback Period** = Battery Cost / Annual Uplift

### Battery Dispatch Logic
```
If Market Price < Daily Average Price:
    - Charge battery with available solar
    - Apply efficiency loss on charge
    - Sell remaining solar to grid

If Market Price > Daily Average Price:
    - Discharge battery to grid
    - Apply efficiency loss on discharge
    - Sell solar + battery energy to grid
```

## Data Validation

### Input Validation
- All sliders have defined ranges and steps
- Dropdown selections are limited to valid options
- Calculated values are checked for reasonableness

### Error Prevention
- Bounds checking for array access
- Null value handling for missing data
- Unit conversion validation
- Efficiency bounds (0-1 range)

## Output Data Structure

### Simulation Results
```python
results = {
    'revenues': {
        'standalone': float,      # Annual standalone solar revenue
        'hybrid': float,         # Annual solar+storage revenue  
        'uplift_dollars': float,  # Additional revenue from storage
        'uplift_percent': float   # Percentage increase
    },
    'kpis': {
        'battery_cost': float,           # Total battery investment
        'payback_period_years': float    # Years to recover investment
    },
    'hourly_data': DataFrame  # 8,760 rows of hourly results
}
```

### Hourly Data Columns
- `price_per_mwh`: Electricity price
- `solar_generation_kw`: Solar output
- `revenue_standalone`: Standalone revenue
- `revenue_hybrid`: Hybrid revenue
- `battery_soc_percent`: Battery state of charge
- `energy_sold_solar`: Energy from solar
- `energy_sold_battery`: Energy from battery

## Common Issues and Solutions

### Issue 1: Index Errors
**Problem:** Array access out of bounds
**Solution:** Bounds checking in simulation loop
**Code:** `solar_gen_kw = solar_profile.iloc[i] if i < len(solar_profile) else 0`

### Issue 2: Pandas Indexing
**Problem:** Modifying pandas Series in place
**Solution:** Convert to numpy array before modification
**Code:** `prices_array = np.array(prices)`

### Issue 3: Missing Price Data
**Problem:** Solar profile generation without price data
**Solution:** Pass price_data parameter or use fallback
**Code:** `generate_solar_profile(peak_mw, price_data=price_data)`

### Issue 4: Division by Zero
**Problem:** Payback calculation with zero uplift
**Solution:** Check for positive values before division
**Code:** `payback = cost / uplift if uplift > 0 else float('inf')`

## Performance Considerations

### Memory Usage
- 8,760 hours × 7 columns = ~60,000 data points
- Typical memory usage: ~50-100 MB
- No memory leaks in simulation loop

### Processing Time
- Full simulation: ~2-5 seconds
- Price data generation: ~1-2 seconds
- Solar profile generation: ~0.5 seconds
- Chart rendering: ~1-2 seconds

## Testing and Validation

### Unit Tests
- Price data generation produces expected ranges
- Solar profile follows sine wave pattern
- Battery dispatch logic works correctly
- Financial calculations are accurate

### Integration Tests
- Full simulation runs without errors
- Results are consistent across runs
- Charts render correctly
- Export functions work

### Edge Cases
- Zero solar generation (night)
- Full battery capacity
- Empty battery
- Extreme price events
- Very low/high efficiency

## Usage Instructions

### Running the Dashboard
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run arbitrage_dashboard.py
```

### Required Python Packages
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.15.0
- requests >= 2.31.0

### Browser Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- Minimum resolution: 1024×768

## Troubleshooting

### Common Errors
1. **ModuleNotFoundError**: Install missing packages
2. **IndexError**: Check data alignment
3. **ValueError**: Verify input ranges
4. **TypeError**: Check data types

### Debug Mode
- Enable Streamlit debug mode
- Check console for error messages
- Verify data types and shapes
- Test individual components

This guide provides complete information about the data inputs and processing for the Solar+Storage Arbitrage Value Calculator.
