import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SolarStorageArbitrage:
    def __init__(self, api_key='pWeHccHjT5QVlxMMoKpKmAKJLsqAGZZM9vdDREYS'):
        """
        Initialize the Solar + Storage Arbitrage calculator.
        
        Note: API key parameter kept for compatibility but not used.
        Price data is generated synthetically based on realistic market patterns.
        """
        self.api_key = api_key
        self.price_data = None
        self.solar_profile = None
        self.results = None

    def load_price_data(self, source_name="California (CAISO)", year=2023):
        """
        Generate realistic synthetic wholesale electricity price data.
        """
        cache_filename = f"cache_synthetic_{source_name.replace(' ', '_')}_{year}.csv"
        
        if os.path.exists(cache_filename):
            print(f"Loading synthetic price data from cache: {cache_filename}")
            self.price_data = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
            
            # FIX: Use set_axis instead of direct index assignment
            if self.price_data.index.tz is None:
                new_index = self.price_data.index.tz_localize('UTC')
                self.price_data = self.price_data.set_axis(new_index)
                
            print(f"Loaded {len(self.price_data)} hours")
            print(f"Price range: ${self.price_data['price_per_mwh'].min():.2f} - ${self.price_data['price_per_mwh'].max():.2f}/MWh")
            print(f"Mean price: ${self.price_data['price_per_mwh'].mean():.2f}/MWh")
            return self.price_data
        
        print(f"Generating synthetic wholesale price data for {source_name}, {year}...")
        print("Based on realistic market patterns: seasonal variation, duck curve, volatility")
        
        # Create hourly index for full year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23)
        hourly_index = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducibility
        
        # Base price patterns by market
        if "California" in source_name:
            base_price = 45  # CAISO typical average
            volatility = 0.3
            duck_curve_strength = 0.4
        else:  # Texas ERCOT
            base_price = 35  # ERCOT typical average
            volatility = 0.25
            duck_curve_strength = 0.3
        
        # Generate prices for each hour
        prices = []
        for timestamp in hourly_index:
            hour = timestamp.hour
            day_of_year = timestamp.timetuple().tm_yday
            
            # Seasonal variation (higher in summer/winter)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Duck curve pattern (low during solar hours, high in evening)
            if 6 <= hour <= 16:  # Solar hours - lower prices
                duck_factor = 0.7 - duck_curve_strength * np.sin(np.pi * (hour - 6) / 10)
            else:  # Evening/night - higher prices
                duck_factor = 1.0 + duck_curve_strength * np.maximum(0, np.sin(np.pi * (hour - 17) / 7))
            
            # Random volatility
            random_factor = 1 + np.random.normal(0, volatility)
            
            # Calculate final price
            price = base_price * seasonal_factor * duck_factor * random_factor
            
            # Ensure positive prices
            price = max(price, 5.0)
            
            prices.append(price)
        
        # Create DataFrame
        self.price_data = pd.DataFrame({'price_per_mwh': prices}, index=hourly_index)
        
        print(f"✓ Generated {len(self.price_data)} hours of synthetic price data")
        print(f"  Price range: ${self.price_data['price_per_mwh'].min():.2f} - ${self.price_data['price_per_mwh'].max():.2f}/MWh")
        print(f"  Mean price: ${self.price_data['price_per_mwh'].mean():.2f}/MWh")
        print(f"  Median price: ${self.price_data['price_per_mwh'].median():.2f}/MWh")
        print(f"  Std deviation: ${self.price_data['price_per_mwh'].std():.2f}/MWh")
        
        # Calculate and display price statistics by time of day
        hourly_avg = self.price_data.groupby(self.price_data.index.hour)['price_per_mwh'].mean()
        print(f"  Average prices: Morning (6-10am): ${hourly_avg[6:10].mean():.2f}, "
            f"Midday (10am-4pm): ${hourly_avg[10:16].mean():.2f}, "
            f"Evening (5-9pm): ${hourly_avg[17:21].mean():.2f}")
        
        # Save to cache
        print(f"Saving to cache: {cache_filename}")
        self.price_data.to_csv(cache_filename)
        
        # FIX: Return the data
        return self.price_data

    
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
        
        print(f"Generating solar profile for {peak_mw} MW system...")
        
        # Create hourly index for the period - match the price data index exactly
        if num_days is None or num_days >= len(self.price_data) // 24:
            # Use the full price data period
            hourly_index = self.price_data.index
        else:
            # Use a subset of the price data period
            start_date = self.price_data.index[0]
            end_date = start_date + timedelta(days=num_days)
            hourly_index = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')
        
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
        
        solar_series = pd.Series(solar_profile, index=hourly_index)
        print(f"Generated {len(solar_series)} hours of solar data")
        print(f"Peak generation: {solar_series.max():.0f} kW")
        print(f"Daily energy: {solar_series.sum() / 1000:.1f} MWh")
        
        return solar_series
    
    def run_simulation(self, solar_profile, params):
        """
        Run the core arbitrage simulation with CORRECTED efficiency calculations.
        
        Args:
            solar_profile (pd.Series): Solar generation profile in kW
            params (dict): Simulation parameters from dashboard
            
        Returns:
            dict: Simulation results including revenues, KPIs, and hourly data
        """
        # Extract parameters
        battery_power_mw = params['battery_power_mw']
        battery_capacity_mwh = params['battery_capacity_mwh']
        battery_capex_per_kwh = params['battery_capex_per_kwh']
        round_trip_efficiency = params['round_trip_efficiency']
        
        # CORRECTED: Split round-trip efficiency into charge and discharge components
        # Based on industry standard: charge_eff * discharge_eff = round_trip_eff
        charge_efficiency = np.sqrt(round_trip_efficiency)
        discharge_efficiency = np.sqrt(round_trip_efficiency)
        
        print(f"Running arbitrage simulation...")
        print(f"   Battery: {battery_power_mw} MW power, {battery_capacity_mwh} MWh capacity")
        print(f"   Round-trip efficiency: {round_trip_efficiency*100:.1f}%")
        print(f"   Charge efficiency: {charge_efficiency*100:.1f}%, Discharge efficiency: {discharge_efficiency*100:.1f}%")
        print(f"   CAPEX: ${battery_capex_per_kwh}/kWh")
        
        # Convert solar profile to MW
        solar_mw = solar_profile / 1000
        
        # Initialize battery variables
        battery_energy_mwh = 0
        
        # Prepare data for simulation
        timestamps = self.price_data.index
        prices = self.price_data['price_per_mwh'].values
        
        # Ensure solar profile is properly aligned with price data
        if len(solar_mw) != len(timestamps):
            # If lengths don't match, reindex to match price data
            solar_values = solar_mw.reindex(timestamps, fill_value=0).values
        else:
            solar_values = solar_mw.values
        
        # Initialize result arrays
        battery_charge_mw = np.zeros(len(timestamps))
        battery_discharge_mw = np.zeros(len(timestamps))
        battery_soc_mwh = np.zeros(len(timestamps))
        grid_sales_mw = np.zeros(len(timestamps))
        grid_purchases_mw = np.zeros(len(timestamps))
        
        # Run arbitrage logic
        for i, timestamp in enumerate(timestamps):
            price = prices[i]
            solar = solar_values[i]
            
            # Calculate price thresholds using quantiles for better optimization
            day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(hours=24)
            
            # Convert timestamps to numpy array for boolean indexing
            timestamps_array = np.array(timestamps)
            day_mask = (timestamps_array >= day_start) & (timestamps_array < day_end)
            day_prices = prices[day_mask]
            
            if len(day_prices) > 0:
                low_threshold = np.quantile(day_prices, 0.33)  # Bottom 33% of prices
                high_threshold = np.quantile(day_prices, 0.67)  # Top 33% of prices
            else:
                low_threshold = price
                high_threshold = price
            
            # Initialize hourly variables
            charge_mw = 0
            discharge_mw = 0
            grid_sales = 0
            grid_purchases = 0
            
            # IMPROVED Arbitrage strategy: charge when cheap, discharge when expensive
            if price < low_threshold and battery_energy_mwh < battery_capacity_mwh:
                # CHARGE: Price is low - charge battery using solar + grid
                available_solar = solar
                remaining_capacity = battery_capacity_mwh - battery_energy_mwh
                
                # Maximum charge power limited by battery power rating and remaining capacity
                max_charge_power = min(battery_power_mw, remaining_capacity / charge_efficiency)
                
                # Use solar first, then grid if needed and profitable
                solar_to_battery = min(available_solar, max_charge_power)
                
                # Only buy from grid if price is very low (bottom 33%)
                grid_to_battery = min(max_charge_power - solar_to_battery, battery_power_mw - solar_to_battery)
                
                charge_mw = solar_to_battery + grid_to_battery
                
                # CORRECTED: Apply charge efficiency when storing energy
                battery_energy_mwh += charge_mw * charge_efficiency
                
                # Grid transactions
                grid_purchases = grid_to_battery
                grid_sales = max(0, solar - solar_to_battery)
                
            elif price > high_threshold and battery_energy_mwh > 0:
                # DISCHARGE: Price is high - discharge battery to grid
                # Maximum discharge limited by battery power and available energy
                max_discharge_power = min(battery_power_mw, battery_energy_mwh)
                
                discharge_mw = max_discharge_power
                
                # CORRECTED: Energy removed from battery (before efficiency losses)
                battery_energy_mwh -= discharge_mw
                
                # CORRECTED: Actual energy to grid (after discharge efficiency losses)
                energy_to_grid = discharge_mw * discharge_efficiency
                
                # Grid transactions
                grid_sales = solar + energy_to_grid
                grid_purchases = 0
                
            else:
                # NO ARBITRAGE: Price in middle range - just sell solar directly
                grid_sales = solar
                grid_purchases = 0
            
            # Store results for this hour
            battery_charge_mw[i] = charge_mw
            battery_discharge_mw[i] = discharge_mw
            battery_soc_mwh[i] = battery_energy_mwh
            grid_sales_mw[i] = grid_sales
            grid_purchases_mw[i] = grid_purchases
        
        # Create results DataFrame
        results = {
            'hourly_data': pd.DataFrame({
                'price_per_mwh': prices,
                'solar_mw': solar_values,
                'battery_charge_mw': battery_charge_mw,
                'battery_discharge_mw': battery_discharge_mw,
                'battery_soc_mwh': battery_soc_mwh,
                'grid_sales_mw': grid_sales_mw,
                'grid_purchases_mw': grid_purchases_mw
            }, index=timestamps),
            'revenues': {},
            'kpis': {}
        }
        
        # Calculate revenues
        # Energy sold per hour: grid_sales_mw (MW) * 1 hour = MWh
        # Revenue per hour: Energy (MWh) * Price ($/MWh) = $
        # Division by 1000 converts revenue to thousands of $ (k$) for cleaner reporting
        results['hourly_data']['revenue'] = (
            results['hourly_data']['grid_sales_mw'] * results['hourly_data']['price_per_mwh'] / 1000
        )
        results['hourly_data']['cost'] = (
            results['hourly_data']['grid_purchases_mw'] * results['hourly_data']['price_per_mwh'] / 1000
        )
        results['hourly_data']['net_revenue'] = results['hourly_data']['revenue'] - results['hourly_data']['cost']
        
        # Calculate total annual revenue with battery
        total_revenue = results['hourly_data']['net_revenue'].sum()
        
        # Calculate solar-only revenue (baseline: selling all solar directly to grid)
        solar_only_revenue = (results['hourly_data']['solar_mw'] * results['hourly_data']['price_per_mwh']).sum()
        
        # Calculate arbitrage uplift (additional revenue from battery storage)
        uplift = total_revenue - solar_only_revenue
        
        # Calculate battery costs
        battery_cost_dollars = battery_capacity_mwh * 1000 * battery_capex_per_kwh
        
        # Annual O&M costs (industry standard: 2.5% of capital cost)
        annual_om_dollars = battery_cost_dollars * 0.025
        
        # Net annual benefit (uplift minus O&M)
        net_annual_benefit = uplift - annual_om_dollars
        
        # Simple payback period
        if net_annual_benefit > 0:
            payback_years = battery_cost_dollars / net_annual_benefit
        else:
            payback_years = float('inf')
        
        results['revenues'] = {
            'total_dollars': total_revenue,
            'solar_only_dollars': solar_only_revenue,
            'uplift_dollars': uplift,
            'uplift_percent': (uplift / solar_only_revenue * 100) if solar_only_revenue > 0 else 0
        }
        
        results['kpis'] = {
            'battery_cost_dollars': battery_cost_dollars,
            'annual_om_dollars': annual_om_dollars,
            'net_annual_benefit_dollars': net_annual_benefit,
            'payback_years': payback_years,
            'total_cycles': results['hourly_data']['battery_discharge_mw'].sum() / battery_capacity_mwh if battery_capacity_mwh > 0 else 0
        }
        
        # Print simulation results summary
        print(f"\nSimulation Results:")
        print(f"  Total revenue (solar+storage): ${total_revenue:,.0f}")
        print(f"  Solar-only revenue: ${solar_only_revenue:,.0f}")
        print(f"  Revenue uplift: ${uplift:,.0f} ({results['revenues']['uplift_percent']:.1f}%)")
        print(f"  Battery cost: ${battery_cost_dollars:,.0f}")
        print(f"  Annual O&M: ${annual_om_dollars:,.0f}")
        print(f"  Net annual benefit: ${net_annual_benefit:,.0f}")
        print(f"  Simple payback: {payback_years:.1f} years")
        print(f"  Total battery cycles: {results['kpis']['total_cycles']:.0f}")
        
        return results
    
    @staticmethod
    def create_arbitrage_chart(results, start_date, num_days=7):
        """
        Create a visualization of the arbitrage strategy.
        
        Args:
            results (dict): Simulation results
            start_date (datetime): Start date for visualization
            num_days (int): Number of days to show
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        # Filter data for the specified period
        if hasattr(start_date, 'date'):
            start_datetime = start_date
        else:
            start_datetime = datetime.combine(start_date, datetime.min.time())
        
        # Ensure timezone consistency
        if results['hourly_data'].index.tz is not None:
            if start_datetime.tzinfo is None:
                start_datetime = start_datetime.replace(tzinfo=None)
                start_datetime = pd.Timestamp(start_datetime).tz_localize('UTC')
            else:
                start_datetime = pd.Timestamp(start_datetime).tz_convert('UTC')
        
        end_date = start_datetime + timedelta(days=num_days)
        mask = (results['hourly_data'].index >= start_datetime) & (results['hourly_data'].index < end_date)
        data = results['hourly_data'][mask].copy()
        
        if data.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Electricity Price ($/MWh)', 
                'Solar Generation (MW)', 
                'Battery Charge/Discharge (MW)',
                'Battery State of Charge (MWh)'
            ),
            vertical_spacing=0.08
        )
        
        # Price chart with threshold lines
        fig.add_trace(
            go.Scatter(x=data.index, y=data['price_per_mwh'], name='Price', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Solar generation
        fig.add_trace(
            go.Scatter(x=data.index, y=data['solar_mw'], name='Solar Generation', 
                      line=dict(color='orange'), fill='tozeroy'),
            row=2, col=1
        )
        
        # Battery charge/discharge
        fig.add_trace(
            go.Bar(x=data.index, y=data['battery_charge_mw'], name='Charging', 
                  marker=dict(color='green')),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=-data['battery_discharge_mw'], name='Discharging', 
                  marker=dict(color='red')),
            row=3, col=1
        )
        
        # Battery state of charge (now using tracked SOC)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['battery_soc_mwh'], name='State of Charge', 
                      line=dict(color='purple', width=2)),
            row=4, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Time", row=4, col=1)
        fig.update_yaxes(title_text="$/MWh", row=1, col=1)
        fig.update_yaxes(title_text="MW", row=2, col=1)
        fig.update_yaxes(title_text="MW", row=3, col=1)
        fig.update_yaxes(title_text="MWh", row=4, col=1)
        
        fig.update_layout(
            height=1000,
            title_text=f"Solar + Storage Arbitrage Strategy: {start_datetime.strftime('%Y-%m-%d')} to {(start_datetime + timedelta(days=num_days-1)).strftime('%Y-%m-%d')}",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig