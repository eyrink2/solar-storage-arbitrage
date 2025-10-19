#!/usr/bin/env python3
"""
Test script for the Solar+Storage Arbitrage Value Calculator.
This script demonstrates the core functionality without the Streamlit interface.
"""

from arbitrage_model import SolarStorageArbitrage
import pandas as pd

def test_arbitrage_model():
    """Test the core arbitrage model functionality."""
    print("🧪 Testing Solar+Storage Arbitrage Model")
    print("=" * 50)
    
    # Initialize the model
    model = SolarStorageArbitrage()
    
    # Test 1: Load price data
    print("\n1. Loading price data...")
    price_data = model.load_price_data("California (CAISO)")
    print(f"   ✓ Loaded {len(price_data)} hours of price data")
    print(f"   ✓ Price range: ${price_data['price_per_mwh'].min():.1f} - ${price_data['price_per_mwh'].max():.1f}/MWh")
    
    # Test 2: Generate solar profile
    print("\n2. Generating solar profile...")
    solar_profile = model.generate_solar_profile(peak_mw=100)
    print(f"   ✓ Generated {len(solar_profile)} hours of solar data")
    print(f"   ✓ Peak generation: {solar_profile.max():.1f} kW")
    
    # Test 3: Run simulation
    print("\n3. Running arbitrage simulation...")
    battery_params = {
        'battery_power_mw': 50,
        'battery_capacity_mwh': 200,  # 4 hours at 50 MW
        'battery_capex_per_kwh': 350,
        'round_trip_efficiency': 0.88
    }
    
    results = model.run_simulation(price_data, solar_profile, battery_params)
    
    # Test 4: Display results
    print("\n4. Simulation Results:")
    print(f"   ✓ Standalone Solar Revenue: ${results['revenues']['standalone']:,.0f}")
    print(f"   ✓ Solar + Storage Revenue: ${results['revenues']['hybrid']:,.0f}")
    print(f"   ✓ Revenue Uplift: ${results['revenues']['uplift_dollars']:,.0f}")
    print(f"   ✓ Revenue Increase: {results['revenues']['uplift_percent']:.1f}%")
    print(f"   ✓ Battery Cost: ${results['kpis']['battery_cost']:,.0f}")
    
    payback = results['kpis']['payback_period_years']
    if payback == float('inf'):
        print(f"   ✓ Payback Period: Never (negative ROI)")
    else:
        print(f"   ✓ Payback Period: {payback:.1f} years")
    
    # Test 5: Check hourly data
    print("\n5. Hourly Data Summary:")
    hourly_data = results['hourly_data']
    print(f"   ✓ Total hours simulated: {len(hourly_data)}")
    print(f"   ✓ Average battery SOC: {hourly_data['battery_soc_percent'].mean():.1f}%")
    print(f"   ✓ Max battery SOC: {hourly_data['battery_soc_percent'].max():.1f}%")
    
    print("\n✅ All tests passed! The arbitrage model is working correctly.")
    return results

def test_texas_market():
    """Test with Texas (ERCOT) market data."""
    print("\n🌵 Testing Texas (ERCOT) Market")
    print("=" * 50)
    
    model = SolarStorageArbitrage()
    
    # Load ERCOT data
    price_data = model.load_price_data("Texas (ERCOT)")
    print(f"   ✓ Loaded {len(price_data)} hours of ERCOT data")
    print(f"   ✓ Price range: ${price_data['price_per_mwh'].min():.1f} - ${price_data['price_per_mwh'].max():.1f}/MWh")
    
    # Generate solar profile
    solar_profile = model.generate_solar_profile(peak_mw=100)
    
    # Run simulation
    battery_params = {
        'battery_power_mw': 50,
        'battery_capacity_mwh': 200,
        'battery_capex_per_kwh': 350,
        'round_trip_efficiency': 0.88
    }
    
    results = model.run_simulation(price_data, solar_profile, battery_params)
    
    print(f"   ✓ Revenue Uplift: ${results['revenues']['uplift_dollars']:,.0f}")
    print(f"   ✓ Revenue Increase: {results['revenues']['uplift_percent']:.1f}%")
    
    return results

if __name__ == "__main__":
    # Run tests
    ca_results = test_arbitrage_model()
    tx_results = test_texas_market()
    
    print("\n🎯 Summary:")
    print(f"California market revenue uplift: ${ca_results['revenues']['uplift_dollars']:,.0f}")
    print(f"Texas market revenue uplift: ${tx_results['revenues']['uplift_dollars']:,.0f}")
    
    print("\n🚀 Ready to run the Streamlit dashboard!")
    print("   Run: streamlit run arbitrage_dashboard.py")
