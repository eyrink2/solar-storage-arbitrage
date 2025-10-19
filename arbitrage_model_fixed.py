import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SolarStorageArbitrage:
    """
    A comprehensive model for calculating the economic value of adding battery storage
    to a solar project through energy arbitrage.
    
    This model demonstrates how battery storage can increase revenue by storing cheap
    midday energy and selling it during expensive evening peaks.
    """
    
    def __init__(self):
        self.price_data = None
        self.solar_profile = None
        self.results = None
        
    def load_price_data(self, source_name="California (CAISO)"):
        """
        Load electricity price data from various sources.
        
        Args:
            source_name (str): Name of the price market to load
            
        Returns:
            pd.DataFrame: Clean price data with datetime index
        """
        if source_name == "California (CAISO)":
            # For demo purposes, create synthetic CAISO-like data
            # In production, this would pull from EIA API or CAISO CSV
            return self._create_synthetic_caiso_data()
        elif source_name == "Texas (ERCOT)":
            # For demo purposes, create synthetic ERCOT-like data
            return self._create_synthetic_ercot_data()
        else:
            raise ValueError(f"Unknown price source: {source_name}")
    
    def _create_synthetic_caiso_data(self):
        """Create synthetic CAISO-like price data with duck curve characteristics."""
        # Create a full year of hourly data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31, 23, 0)
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Base price with seasonal variation
        seasonal_base = 50 + 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / (365.25 * 24))
        
        # Daily duck curve pattern (high morning/evening, low midday)
        hour_of_day = date_range.hour
        daily_pattern = 30 + 40 * np.sin(np.pi * (hour_of_day - 6) / 12) + \
                       20 * np.sin(np.pi * (hour_of_day - 18) / 6)
        
        # Add some randomness
        noise = np.random.normal(0, 5, len(date_range))
        
        # Combine all effects
        prices = seasonal_base + daily_pattern + noise
        
        # Ensure positive prices
        prices = np.maximum(prices, 10)
        
        # Create some extreme price events (like heat waves)
        extreme_days = np.random.choice(len(date_range), size=20, replace=False)
        for day in extreme_days:
            if 14 <= date_range[day].hour <= 20:  # Peak hours
                prices[day] *= np.random.uniform(2, 4)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'price_per_mwh': prices
        }).set_index('timestamp')
    
    def _create_synthetic_ercot_data(self):
        """Create synthetic ERCOT-like price data with high volatility."""
        # Create a full year of hourly data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31, 23, 0)
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Base price with seasonal variation
        seasonal_base = 45 + 25 * np.sin(2 * np.pi * np.arange(len(date_range)) / (365.25 * 24))
        
        # Daily pattern with higher volatility
        hour_of_day = date_range.hour
        daily_pattern = 25 + 30 * np.sin(np.pi * (hour_of_day - 6) / 12) + \
                       15 * np.sin(np.pi * (hour_of_day - 18) / 6)
        
        # Higher volatility for ERCOT
        noise = np.random.normal(0, 8, len(date_range))
        
        # Combine all effects
        prices = seasonal_base + daily_pattern + noise
        
        # Ensure positive prices
        prices = np.maximum(prices, 5)
        
        # Create more extreme price events (ERCOT is known for volatility)
        extreme_days = np.random.choice(len(date_range), size=50, replace=False)
        for day in extreme_days:
            if 14 <= date_range[day].hour <= 20:  # Peak hours
                prices[day] *= np.random.uniform(3, 8)
        
        return pd.DataFrame({
            'timestamp': date_range,
            'price_per_mwh': prices
        }).set_index('timestamp')
    
    def generate_solar_profile(self, peak_mw, num_days=None):
        """
        Generate synthetic solar generation profile.
        
        Args:
            peak_mw (float): Peak solar capacity in MW
            num_days (int): Number of days to generate (default: match price data)
            
        Returns:
            pd.Series: Hourly solar generation in kW
        """
        if num_days is None:
            num_days = len(self.price_data) // 24
        
        # Create hourly index for the period
        start_date = self.price_data.index[0]
        end_date = start_date + timedelta(days=num_days)
        hourly_index = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Generate solar profile using sine wave
        hour_of_day = hourly_index.hour
        solar_profile = []
        
        for hour in hour_of_day:
            if 6 <= hour <= 18:  # Daylight hours
                # Sine wave that peaks at noon
                output_kw = max(0, np.sin(np.pi * (hour - 6) / 12)) * peak_mw * 1000
            else:
                output_kw = 0
            
            solar_profile.append(output_kw)
        
        return pd.Series(solar_profile, index=hourly_index)
    
    def run_simulation(self, price_data, solar_profile, battery_params):
        """
        Run the core arbitrage simulation.
        
        Args:
            price_data (pd.DataFrame): Price data with 'price_per_mwh' column
            solar_profile (pd.Series): Solar generation profile in kW
            battery_params (dict): Battery system parameters
            
        Returns:
            dict: Simulation results including revenues and KPIs
        """
        # Extract battery parameters
        battery_power_mw = battery_params['battery_power_mw']
        battery_capacity_mwh = battery_params['battery_capacity_mwh']
        battery_capex_per_kwh = battery_params['battery_capex_per_kwh']
        round_trip_efficiency = battery_params['round_trip_efficiency']
        
        # Convert to consistent units
        battery_power_kw = battery_power_mw * 1000
        battery_capacity_kwh = battery_capacity_mwh * 1000
        round_trip_efficiency_sqrt = np.sqrt(round_trip_efficiency)
        
        # Initialize results
        results = {
            'hourly_data': pd.DataFrame(index=price_data.index),
            'revenues': {},
            'kpis': {}
        }
        
        # Calculate daily average prices for dispatch decisions
        daily_avg_prices = price_data.groupby(price_data.index.date)['price_per_mwh'].mean()
        
        # Initialize battery state
        current_charge_kwh = 0
        hourly_revenues_standalone = []
        hourly_revenues_hybrid = []
        battery_soc = []
        energy_sold_solar = []
        energy_sold_battery = []
        
        # Main simulation loop
        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            # Get inputs for this hour
            solar_gen_kw = solar_profile.iloc[i] if i < len(solar_profile) else 0
            market_price_per_mwh = row['price_per_mwh']
            market_price_per_kwh = market_price_per_mwh / 1000
            
            # Get daily average price for dispatch decision
            date_key = timestamp.date()
            daily_avg_price = daily_avg_prices.get(date_key, market_price_per_mwh)
            
            # Calculate standalone solar revenue
            standalone_revenue = solar_gen_kw * market_price_per_kwh
            hourly_revenues_standalone.append(standalone_revenue)
            
            # Storage dispatch logic
            if market_price_per_mwh < daily_avg_price:
                # Charge signal: store energy when prices are low
                energy_available_to_charge = solar_gen_kw
                energy_to_charge = min(
                    energy_available_to_charge,
                    battery_power_kw,
                    battery_capacity_kwh - current_charge_kwh
                )
                
                # Update battery state (apply half efficiency loss on charge)
                current_charge_kwh += energy_to_charge * round_trip_efficiency_sqrt
                
                # Energy sold to grid
                energy_sold_to_grid = solar_gen_kw - energy_to_charge
                energy_sold_solar.append(energy_sold_to_grid)
                energy_sold_battery.append(0)
                
            else:
                # Discharge signal: sell stored energy when prices are high
                energy_to_discharge = min(battery_power_kw, current_charge_kwh)
                
                # Update battery state
                current_charge_kwh -= energy_to_discharge
                
                # Energy sold to grid (apply other half of efficiency loss on discharge)
                energy_sold_to_grid = solar_gen_kw + (energy_to_discharge / round_trip_efficiency_sqrt)
                energy_sold_solar.append(solar_gen_kw)
                energy_sold_battery.append(energy_to_discharge / round_trip_efficiency_sqrt)
            
            # Calculate hybrid revenue
            hybrid_revenue = energy_sold_to_grid * market_price_per_kwh
            hourly_revenues_hybrid.append(hybrid_revenue)
            
            # Track battery state of charge
            battery_soc.append(current_charge_kwh / battery_capacity_kwh * 100)
        
        # Calculate total revenues
        total_revenue_standalone = sum(hourly_revenues_standalone)
        total_revenue_hybrid = sum(hourly_revenues_hybrid)
        
        # Calculate financial KPIs
        revenue_uplift_dollars = total_revenue_hybrid - total_revenue_standalone
        revenue_uplift_percent = (revenue_uplift_dollars / total_revenue_standalone) * 100 if total_revenue_standalone > 0 else 0
        
        total_battery_cost = battery_capacity_kwh * battery_capex_per_kwh
        simple_payback_period_years = total_battery_cost / revenue_uplift_dollars if revenue_uplift_dollars > 0 else float('inf')
        
        # Store results
        results['revenues'] = {
            'standalone': total_revenue_standalone,
            'hybrid': total_revenue_hybrid,
            'uplift_dollars': revenue_uplift_dollars,
            'uplift_percent': revenue_uplift_percent
        }
        
        results['kpis'] = {
            'battery_cost': total_battery_cost,
            'payback_period_years': simple_payback_period_years
        }
        
        # Store hourly data for visualization
        results['hourly_data'] = pd.DataFrame({
            'price_per_mwh': price_data['price_per_mwh'],
            'solar_generation_kw': solar_profile,
            'revenue_standalone': hourly_revenues_standalone,
            'revenue_hybrid': hourly_revenues_hybrid,
            'battery_soc_percent': battery_soc,
            'energy_sold_solar': energy_sold_solar,
            'energy_sold_battery': energy_sold_battery
        }, index=price_data.index)
        
        return results
    
    def create_arbitrage_chart(self, results, start_date=None, num_days=1):
        """
        Create the main arbitrage visualization showing price and energy dispatch.
        
        Args:
            results (dict): Simulation results
            start_date (str): Start date for the chart (default: first day)
            num_days (int): Number of days to show
            
        Returns:
            plotly.graph_objects.Figure: The arbitrage chart
        """
        hourly_data = results['hourly_data']
        
        # Select date range
        if start_date is None:
            start_date = hourly_data.index[0]
        else:
            start_date = pd.to_datetime(start_date)
        
        end_date = start_date + timedelta(days=num_days)
        chart_data = hourly_data.loc[start_date:end_date]
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Energy Arbitrage Dispatch', 'Battery State of Charge'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Add price line (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data['price_per_mwh'],
                name='Electricity Price',
                line=dict(color='red', width=2),
                yaxis='y'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add energy bars (right y-axis)
        fig.add_trace(
            go.Bar(
                x=chart_data.index,
                y=chart_data['energy_sold_solar'],
                name='Energy from Solar',
                marker_color='gold',
                opacity=0.8,
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Bar(
                x=chart_data.index,
                y=chart_data['energy_sold_battery'],
                name='Energy from Battery',
                marker_color='blue',
                opacity=0.8,
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add battery SOC (second subplot)
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data['battery_soc_percent'],
                name='Battery SOC',
                line=dict(color='green', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Solar+Storage Arbitrage Example ({num_days} Day{'s' if num_days > 1 else ''})",
            height=600,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Energy (kW)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Battery SOC (%)", row=2, col=1)
        
        return fig
