import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        cache_filename = f"cache_synthetic_{source_name.replace(' ', '_').replace('(', '').replace(')', '')}_{year}.csv"
        
        if os.path.exists(cache_filename):
            print(f"Loading synthetic price data from cache: {cache_filename}")
            try:
                self.price_data = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
            except (ValueError, KeyError):
                # Cache file is old format, delete and regenerate
                print("  ⚠️  Old cache format detected, regenerating...")
                os.remove(cache_filename)
                return self.load_price_data(source_name, year)
            
            # Ensure UTC timezone
            if self.price_data.index.tz is None:
                self.price_data.index = self.price_data.index.tz_localize('UTC')
            else:
                self.price_data.index = self.price_data.index.tz_convert('UTC')
                
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
                duck_factor = 1.0 + duck_curve_strength * max(0, np.sin(np.pi * (hour - 17) / 7))
            
            # Random volatility
            random_factor = 1 + np.random.normal(0, volatility)
            
            # Calculate final price
            price = base_price * seasonal_factor * duck_factor * random_factor
            
            # Ensure positive prices
            price = max(price, 5.0)
            
            prices.append(price)
        
        # Create DataFrame
        self.price_data = pd.DataFrame({'price_per_mwh': prices}, index=hourly_index)
        self.price_data.index.name = 'timestamp'  # Name the index so CSV saves correctly
        
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
        
        return self.price_data

    
    def generate_solar_profile(self, peak_mw, num_days=None):
        """
        Generate synthetic solar generation profile that matches price data index.
        
        Args:
            peak_mw (float): Peak solar capacity in MW
            num_days (int): Number of days to generate (default: match price data)
            
        Returns:
            pd.Series: Hourly solar generation in kW
        """
        if self.price_data is None:
            raise ValueError("Price data must be loaded before generating solar profile")
        
        print(f"Generating solar profile for {peak_mw} MW system...")
        
        # Use the exact index from price data to ensure perfect alignment
        hourly_index = self.price_data.index
        
        # Generate solar profile using sine wave
        solar_profile = []
        
        for timestamp in hourly_index:
            hour = timestamp.hour
            if 6 <= hour <= 18:  # Daylight hours
                # Sine wave that peaks at noon
                output_kw = max(0, np.sin(np.pi * (hour - 6) / 12)) * peak_mw * 1000
            else:
                output_kw = 0
            
            solar_profile.append(output_kw)
        
        solar_series = pd.Series(solar_profile, index=hourly_index)
        self.solar_profile = solar_series
        
        print(f"Generated {len(solar_series)} hours of solar data")
        print(f"Peak generation: {solar_series.max():.0f} kW")
        total_energy = solar_series.sum() / 1000
        print(f"Total annual energy: {total_energy:.1f} MWh")
        
        return solar_series
    
    def run_simulation(self, solar_profile, params):
        """
        Run the core arbitrage simulation with corrected efficiency calculations.
        
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
        
        # Split round-trip efficiency into charge and discharge components
        charge_efficiency = np.sqrt(round_trip_efficiency)
        discharge_efficiency = np.sqrt(round_trip_efficiency)
        
        print(f"\nRunning arbitrage simulation...")
        print(f"   Battery: {battery_power_mw} MW power, {battery_capacity_mwh} MWh capacity")
        print(f"   Round-trip efficiency: {round_trip_efficiency*100:.1f}%")
        print(f"   Charge efficiency: {charge_efficiency*100:.1f}%, Discharge efficiency: {discharge_efficiency*100:.1f}%")
        print(f"   CAPEX: ${battery_capex_per_kwh}/kWh")
        
        # Convert solar profile to MW and ensure alignment
        solar_mw = solar_profile / 1000
        
        # Ensure solar profile matches price data index exactly
        solar_mw = solar_mw.reindex(self.price_data.index, fill_value=0)
        
        # Initialize battery state
        battery_energy_mwh = 0
        
        # Get aligned data
        timestamps = self.price_data.index
        prices = self.price_data['price_per_mwh'].values
        solar_values = solar_mw.values
        
        # Initialize result arrays
        n_hours = len(timestamps)
        battery_charge_mw = np.zeros(n_hours)
        battery_discharge_mw = np.zeros(n_hours)
        battery_soc_mwh = np.zeros(n_hours)
        grid_sales_mw = np.zeros(n_hours)
        grid_purchases_mw = np.zeros(n_hours)
        
        # Run arbitrage logic
        for i in range(n_hours):
            price = prices[i]
            solar = solar_values[i]
            
            # Calculate daily price thresholds
            day_start = timestamps[i].replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(hours=24)
            
            day_mask = (timestamps >= day_start) & (timestamps < day_end)
            day_prices = prices[day_mask]
            
            if len(day_prices) > 0:
                low_threshold = np.percentile(day_prices, 33)
                high_threshold = np.percentile(day_prices, 67)
            else:
                low_threshold = price
                high_threshold = price
            
            # Arbitrage strategy
            if price < low_threshold and battery_energy_mwh < battery_capacity_mwh:
                # CHARGE: Low price - charge battery
                available_solar = solar
                remaining_capacity = battery_capacity_mwh - battery_energy_mwh
                
                max_charge_power = min(battery_power_mw, remaining_capacity / charge_efficiency)
                
                solar_to_battery = min(available_solar, max_charge_power)
                grid_to_battery = min(max_charge_power - solar_to_battery, battery_power_mw - solar_to_battery)
                
                charge_mw = solar_to_battery + grid_to_battery
                battery_energy_mwh += charge_mw * charge_efficiency
                
                grid_purchases_mw[i] = grid_to_battery
                grid_sales_mw[i] = max(0, solar - solar_to_battery)
                battery_charge_mw[i] = charge_mw
                
            elif price > high_threshold and battery_energy_mwh > 0:
                # DISCHARGE: High price - discharge battery
                max_discharge_power = min(battery_power_mw, battery_energy_mwh)
                
                discharge_mw = max_discharge_power
                battery_energy_mwh -= discharge_mw
                
                energy_to_grid = discharge_mw * discharge_efficiency
                
                grid_sales_mw[i] = solar + energy_to_grid
                battery_discharge_mw[i] = discharge_mw
                
            else:
                # NO ARBITRAGE: Sell solar directly
                grid_sales_mw[i] = solar
            
            # Track battery state
            battery_soc_mwh[i] = battery_energy_mwh
        
        # Create results DataFrame
        hourly_data = pd.DataFrame({
            'price_per_mwh': prices,
            'solar_mw': solar_values,
            'battery_charge_mw': battery_charge_mw,
            'battery_discharge_mw': battery_discharge_mw,
            'battery_soc_mwh': battery_soc_mwh,
            'grid_sales_mw': grid_sales_mw,
            'grid_purchases_mw': grid_purchases_mw
        }, index=timestamps)
        
        # Calculate revenues in DOLLARS (not thousands)
        # Revenue = Energy (MWh) × Price ($/MWh) = $
        # Since we're using hourly data: MW × 1 hour = MWh
        hourly_data['revenue'] = hourly_data['grid_sales_mw'] * hourly_data['price_per_mwh']
        hourly_data['cost'] = hourly_data['grid_purchases_mw'] * hourly_data['price_per_mwh']
        hourly_data['net_revenue'] = hourly_data['revenue'] - hourly_data['cost']
        
        # Calculate total revenues (in dollars)
        total_revenue = hourly_data['net_revenue'].sum()
        solar_only_revenue = (hourly_data['solar_mw'] * hourly_data['price_per_mwh']).sum()
        uplift = total_revenue - solar_only_revenue
        
        # Calculate costs
        battery_cost_dollars = battery_capacity_mwh * 1000 * battery_capex_per_kwh
        annual_om_dollars = battery_cost_dollars * 0.025
        net_annual_benefit = uplift - annual_om_dollars
        
        payback_years = battery_cost_dollars / net_annual_benefit if net_annual_benefit > 0 else float('inf')
        
        results = {
            'hourly_data': hourly_data,
            'revenues': {
                'total_dollars': total_revenue,
                'solar_only_dollars': solar_only_revenue,
                'uplift_dollars': uplift,
                'uplift_percent': (uplift / solar_only_revenue * 100) if solar_only_revenue > 0 else 0
            },
            'kpis': {
                'battery_cost_dollars': battery_cost_dollars,
                'annual_om_dollars': annual_om_dollars,
                'net_annual_benefit_dollars': net_annual_benefit,
                'payback_years': payback_years,
                'total_cycles': battery_discharge_mw.sum() / battery_capacity_mwh if battery_capacity_mwh > 0 else 0
            }
        }
        
        print(f"\n✓ Simulation Results:")
        print(f"  Total revenue (solar+storage): ${total_revenue:,.0f}")
        print(f"  Solar-only revenue: ${solar_only_revenue:,.0f}")
        print(f"  Revenue uplift: ${uplift:,.0f} ({results['revenues']['uplift_percent']:.1f}%)")
        print(f"  Battery cost: ${battery_cost_dollars:,.0f}")
        print(f"  Net annual benefit: ${net_annual_benefit:,.0f}")
        print(f"  Simple payback: {payback_years:.1f} years")
        print(f"  Battery cycles per year: {results['kpis']['total_cycles']:.0f}")
        
        return results
    
    @staticmethod
    def create_arbitrage_chart(results, start_date, num_days=7):
        """
        Create a visualization of the arbitrage strategy.
        
        Args:
            results (dict): Simulation results
            start_date (datetime or date): Start date for visualization
            num_days (int): Number of days to show
            
        Returns:
            plotly.graph_objects.Figure: Interactive chart
        """
        # Convert start_date to timezone-aware datetime
        if isinstance(start_date, datetime):
            start_datetime = start_date
        else:
            start_datetime = datetime.combine(start_date, datetime.min.time())
        
        # Make timezone-aware
        if start_datetime.tzinfo is None:
            start_datetime = pd.Timestamp(start_datetime).tz_localize('UTC')
        else:
            start_datetime = pd.Timestamp(start_datetime).tz_convert('UTC')
        
        end_datetime = start_datetime + timedelta(days=num_days)
        
        # Filter data
        mask = (results['hourly_data'].index >= start_datetime) & (results['hourly_data'].index < end_datetime)
        data = results['hourly_data'][mask].copy()
        
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available for selected date range", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
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
        
        # Price chart
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
        
        # Battery state of charge
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
            title_text=f"Solar + Storage Arbitrage: {start_datetime.strftime('%Y-%m-%d')} ({num_days} day{'s' if num_days > 1 else ''})",
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig