import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arbitrage_model import SolarStorageArbitrage
from datetime import datetime, timedelta

def create_sidebar_controls():
    """
    Create the sidebar control panel with all required inputs.
    
    REQUIRED INPUTS FOR SIMULATION:
    1. Price Market: California (CAISO) or Texas (ERCOT)
    2. Solar System Size: 10-500 MW (peak capacity)
    3. Battery Power: 5-250 MW (charge/discharge rate)
    4. Battery Duration: 2-6 hours (energy capacity = power × duration)
    5. Battery CAPEX: $200-600/kWh (capital cost)
    6. Round-Trip Efficiency: 80-95% (energy conversion efficiency)
    """
    st.sidebar.header("Simulation Inputs")
    
    # Data Selection
    st.sidebar.subheader("Price Market")
    price_market = st.sidebar.selectbox(
        "Select Price Market",
        ["California (CAISO)", "Texas (ERCOT)"],
        help="Choose the electricity market for price data"
    )
    
    # System Configuration
    st.sidebar.subheader("System Configuration")
    solar_size_mw = st.sidebar.slider(
        "Solar System Size (MW)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Peak capacity of the solar farm"
    )
    
    battery_power_mw = st.sidebar.slider(
        "Battery Power (MW)",
        min_value=5,
        max_value=250,
        value=50,
        step=5,
        help="Maximum charge/discharge power of the battery"
    )
    
    battery_duration_hours = st.sidebar.slider(
        "Battery Duration (Hours)",
        min_value=2,
        max_value=6,
        value=4,
        step=1,
        help="How long the battery can discharge at full power"
    )
    
    # CALCULATE DERIVED PARAMETER: Battery capacity from power and duration
    # This is a key calculation: Energy = Power × Time
    battery_capacity_mwh = battery_power_mw * battery_duration_hours
    
    # Financial Assumptions
    st.sidebar.subheader("Financial Assumptions")
    battery_capex_per_kwh = st.sidebar.slider(
        "Battery CAPEX ($/kWh)",
        min_value=200,
        max_value=600,
        value=350,
        step=25,
        help="Capital cost per kWh of battery capacity"
    )
    
    round_trip_efficiency = st.sidebar.slider(
        "Round-Trip Efficiency (%)",
        min_value=80,
        max_value=95,
        value=88,
        step=1,
        help="Overall efficiency of charging and discharging"
    ) / 100  # Convert to decimal
    
    # Action Button
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button(
        "Run Simulation",
        type="primary",
        use_container_width=True
    )
    
    return {
        'price_market': price_market,
        'solar_size_mw': solar_size_mw,
        'battery_power_mw': battery_power_mw,
        'battery_capacity_mwh': battery_capacity_mwh,
        'battery_capex_per_kwh': battery_capex_per_kwh,
        'round_trip_efficiency': round_trip_efficiency,
        'run_simulation': run_simulation
    }

def display_kpi_metrics(results):
    """Display the key performance indicators."""
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Annual Revenue Uplift",
            f"${results['revenues']['uplift_dollars']:,.0f}",
            help="Additional annual revenue from adding battery storage"
        )
    
    with col2:
        st.metric(
            "Revenue Increase (%)",
            f"{results['revenues']['uplift_percent']:.1f}%",
            help="Percentage increase in annual revenue"
        )

def display_arbitrage_chart(results):
    """Display the main arbitrage visualization."""
    st.subheader("Example 24-Hour Dispatch Profile")
    
    # Date selector for the chart
    hourly_data = results['hourly_data']
    available_dates = pd.Series(hourly_data.index.date).unique()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.selectbox(
            "Select Date",
            available_dates[:7],  # Show first 7 days
            help="Choose which day to visualize"
        )
    
    with col2:
        num_days = st.slider(
            "Number of Days to Show",
            min_value=1,
            max_value=7,
            value=1,
            help="How many consecutive days to display"
        )
    
    # Create the chart
    model = SolarStorageArbitrage()
    fig = model.create_arbitrage_chart(
        results, 
        start_date=selected_date, 
        num_days=num_days
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.info("""
    **How to Read This Chart:**
    - **Red Line**: Electricity price ($/MWh) - shows when energy is cheap vs expensive
    - **Gold Bars**: Energy sold directly from solar panels
    - **Blue Bars**: Energy discharged from battery storage
    - **Green Line**: Battery state of charge (%) - shows when battery is charging vs discharging
    
    The arbitrage strategy charges the battery when prices are low (midday) and discharges when prices are high (evening peaks).
    """)

def display_battery_activity_chart(results):
    """Display battery state of charge over time."""
    st.subheader("Battery State of Charge")
    
    hourly_data = results['hourly_data']
    
    # Create a simple line chart for battery SOC
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_data.index,
        y=hourly_data['battery_soc_percent'],
        name='Battery SOC',
        line=dict(color='green', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig.update_layout(
        title="Battery State of Charge Over Time",
        xaxis_title="Time",
        yaxis_title="Battery SOC (%)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_revenue_comparison(results):
    """Display revenue comparison between standalone and hybrid systems."""
    st.subheader("Revenue Comparison")
    
    revenues = results['revenues']
    
    # Create comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Standalone Solar',
        x=['Annual Revenue'],
        y=[revenues['standalone']],
        marker_color='gold',
        text=[f"${revenues['standalone']:,.0f}"],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Solar + Storage',
        x=['Annual Revenue'],
        y=[revenues['hybrid']],
        marker_color='blue',
        text=[f"${revenues['hybrid']:,.0f}"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Annual Revenue Comparison",
        yaxis_title="Revenue ($)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Standalone Solar Revenue",
            f"${revenues['standalone']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Solar + Storage Revenue",
            f"${revenues['hybrid']:,.0f}"
        )

def display_system_summary(params, results):
    """Display system configuration summary."""
    st.subheader("System Configuration Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Solar Capacity", f"{params['solar_size_mw']} MW")
        st.metric("Battery Power", f"{params['battery_power_mw']} MW")
    
    with col2:
        st.metric("Battery Capacity", f"{params['battery_capacity_mwh']:.0f} MWh")
        st.metric("Battery Duration", f"{params['battery_capacity_mwh']/params['battery_power_mw']:.1f} hours")
    
    with col3:
        st.metric("Battery CAPEX", f"${params['battery_capex_per_kwh']}/kWh")
        st.metric("Round-Trip Efficiency", f"{params['round_trip_efficiency']*100:.0f}%")

def main():
    """Main function to run the Solar + Storage Arbitrage Value Calculator."""
    # Page configuration
    st.set_page_config(
        page_title="Solar + Storage Arbitrage Value Calculator",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("Solar + Storage Arbitrage Value Calculator")
    st.markdown("""
    This tool demonstrates how adding battery storage to a solar farm can increase revenue by storing 
    cheap midday energy and selling it during expensive evening peaks.
    """)
    
    # Get user inputs
    params = create_sidebar_controls()
    
    # SIMULATION EXECUTION: Run when user clicks button
    if params['run_simulation']:
        with st.spinner("Running simulation..."):
            # STEP 1: Initialize the arbitrage model
            model = SolarStorageArbitrage()
            
            # STEP 2: Load electricity price data (synthetic CAISO/ERCOT)
            # This creates 8,760 hours of realistic price data
            price_data = model.load_price_data(params['price_market'])
            
            # STEP 3: Generate solar generation profile
            # Creates hourly solar output using sine wave (peaks at noon)
            # BUG FIX: Pass price_data to ensure matching time ranges
            solar_profile = model.generate_solar_profile(params['solar_size_mw'], price_data=price_data)
            
            # STEP 4: Prepare battery system parameters
            # All inputs are validated and converted to proper units
            battery_params = {
                'battery_power_mw': params['battery_power_mw'],
                'battery_capacity_mwh': params['battery_capacity_mwh'],
                'battery_capex_per_kwh': params['battery_capex_per_kwh'],
                'round_trip_efficiency': params['round_trip_efficiency']
            }
            
            # STEP 5: Run the core arbitrage simulation
            # This processes all 8,760 hours and calculates revenues
            results = model.run_simulation(price_data, solar_profile, battery_params)
            
            # STEP 6: Store results for display
            # Uses Streamlit session state to persist data
            st.session_state['simulation_results'] = results
            st.session_state['simulation_params'] = params
    
    # Display results if available
    if 'simulation_results' in st.session_state:
        results = st.session_state['simulation_results']
        params = st.session_state['simulation_params']
        
        # Display KPIs
        display_kpi_metrics(results)
        
        st.markdown("---")
        
        # Display system summary
        display_system_summary(params, results)
        
        st.markdown("---")
        
        # Display main visualizations
        display_arbitrage_chart(results)
        
        st.markdown("---")
        
        # Display battery activity
        display_battery_activity_chart(results)
        
        st.markdown("---")
        
        # Display revenue comparison
        display_revenue_comparison(results)
        
        # Add download option
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Results")
        
        if st.sidebar.button("Download Hourly Data"):
            hourly_data = results['hourly_data']
            csv = hourly_data.to_csv()
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"arbitrage_results_{params['solar_size_mw']}MW_{params['battery_power_mw']}MW.csv",
                mime="text/csv"
            )
    
    else:
        # Show instructions when no simulation has been run
        st.info("Use the controls in the sidebar to configure your solar + storage system and click 'Run Simulation' to see the results.")
        
        # Add some educational content
        st.markdown("---")
        st.subheader("How Energy Arbitrage Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **The Problem:**
            - Solar panels generate most electricity during midday
            - Electricity prices are often lowest during midday
            - Peak demand and highest prices occur in the evening
            - Without storage, solar farms miss the high-value evening market
            """)
        
        with col2:
            st.markdown("""
            **The Solution:**
            - Battery storage captures midday solar generation
            - Store energy when prices are low
            - Discharge during evening peaks when prices are high
            - Increase revenue through price arbitrage
            """)
        
        st.markdown("---")
        st.subheader("Key Metrics Explained")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Revenue Uplift**
            - Additional annual revenue from adding storage
            - Measured in dollars per year
            """)
        
        with col2:
            st.markdown("""
            **Revenue Increase %**
            - Percentage improvement over standalone solar
            - Shows relative value of storage addition
            """)
        
        with col3:
            st.markdown("""
            **Payback Period**
            - Years to recover battery investment
            - Key metric for investment decision
            """)

if __name__ == "__main__":
    main()
