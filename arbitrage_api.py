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
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("EIA API key not provided.")
        
        self.price_data = None
        self.solar_profile = None
        self.results = None

    def _fetch_eia_data_v2(self, respondent, start_date, end_date):
        """
        Fetch data using the modern v2 API route for electricity/rto data.
        """
        # Modern v2 route for real-time electricity prices
        url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        
        params = {
            'api_key': self.api_key,
            'frequency': 'hourly',
            'data[]': 'value',
            'facets[respondent][]': respondent,
            # Try 'DF' for Day-Ahead Forecast or use the daily endpoint instead
            # The region-data endpoint may not have hourly LMP data
            # Common types: 'D' = Demand, 'NG' = Net Generation, 'TI' = Total Interchange
            # NOTE: The hourly region-data endpoint may NOT have price data
            # We may need to use a different endpoint entirely
            'facets[type][]': 'D',  # Using demand temporarily to diagnose
            'start': start_date.strftime('%Y-%m-%dT%H'),
            'end': end_date.strftime('%Y-%m-%dT%H'),
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': 0,
            'length': 5000  # Max per request
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            json_data = response.json()
            
            # Check for API errors
            if 'error' in json_data:
                print(f"API Error: {json_data['error']}")
                return pd.DataFrame()
            
            # V2 API structure
            data = json_data.get('response', {}).get('data', [])
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Print the actual data we're getting for debugging
            if len(df) > 0:
                print(f"    DEBUG: Received {len(df)} rows")
                print(f"    DEBUG: Columns: {df.columns.tolist()}")
                if 'type' in df.columns:
                    print(f"    DEBUG: Type values: {df['type'].unique()}")
                if 'value' in df.columns:
                    print(f"    DEBUG: Value range: {df['value'].min()} - {df['value'].max()}")
            
            df = df.rename(columns={'period': 'timestamp', 'value': 'price_per_mwh'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df['price_per_mwh'] = pd.to_numeric(df['price_per_mwh'], errors='coerce')
            df = df.dropna()
            
            # Localize to UTC
            if df.index.tz is None:
                df = df.tz_localize('UTC')
            
            return df[['price_per_mwh']]
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return pd.DataFrame()

    def _fetch_eia_daily_prices(self, region, start_date, end_date):
        """
        Fetch DAILY wholesale price data from EIA API v2.
        Note: This is DAILY data, we'll interpolate to hourly with realistic patterns.
        """
        url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        
        params = {
            'api_key': self.api_key,
            'frequency': 'daily',
            'data[]': 'value',
            'facets[respondent][]': region,
            'facets[type][]': 'DP',  # DP = Day-Ahead Weighted Average Price
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': 0,
            'length': 5000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            
            if 'error' in json_data:
                print(f"API Error: {json_data['error']}")
                return pd.DataFrame()
            
            data = json_data.get('response', {}).get('data', [])
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df = df.rename(columns={'period': 'date', 'value': 'daily_price'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['daily_price'] = pd.to_numeric(df['daily_price'], errors='coerce')
            df = df.dropna()
            
            return df[['daily_price']]
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return pd.DataFrame()

    def load_price_data(self, source_name="California (CAISO)", year=2023):
        """
        Load REAL wholesale price data from EIA and interpolate to hourly with realistic patterns.
        """
        respondent_map = {
            "California (CAISO)": "CISO",
            "Texas (ERCOT)": "ERCO"
        }
        
        respondent = respondent_map.get(source_name)
        if not respondent:
            raise ValueError(f"Unknown price source: {source_name}")
        
        cache_filename = f"cache_prices_{respondent}_{year}.csv"
        
        if os.path.exists(cache_filename):
            print(f"Loading from cache: {cache_filename}")
            self.price_data = pd.read_csv(cache_filename, index_col='timestamp', parse_dates=True)
            if self.price_data.index.tz is None:
                self.price_data.index = self.price_data.index.tz_localize('UTC')
            return self.price_data

        print(f"Fetching REAL wholesale price data from EIA for {source_name}, {year}...")
        
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        
        daily_prices = self._fetch_eia_daily_prices(respondent, start, end)
        
        if daily_prices.empty:
            print("Failed to fetch data from EIA API")
            self.price_data = pd.DataFrame()
            return self.price_data
        
        print(f"Fetched {len(daily_prices)} days of price data")
        print(f"Daily price range: ${daily_prices['daily_price'].min():.2f} - ${daily_prices['daily_price'].max():.2f}/MWh")
        
        # Interpolate daily prices to hourly using duck curve pattern
        hourly_index = pd.date_range(start=start, end=end + timedelta(days=1), freq='h', tz='UTC')[:-1]
        
        # Create hourly dataframe
        hourly_prices = pd.DataFrame(index=hourly_index)
        hourly_prices['date'] = hourly_prices.index.date
        
        # Merge with daily prices
        hourly_prices = hourly_prices.merge(
            daily_prices, left_on='date', right_index=True, how='left'
        )
        
        # Apply hourly pattern (duck curve - higher in evening, lower during solar hours)
        hours = hourly_prices.index.hour
        hourly_multiplier = np.where(
            (hours >= 6) & (hours <= 16),  # Solar hours  
            0.75 + 0.15 * np.sin(np.pi * (hours - 6) / 10),  # Lower during day
            1.0 + 0.4 * np.maximum(0, np.sin(np.pi * (hours - 17) / 7))  # Peak in evening
        )
        
        hourly_prices['price_per_mwh'] = hourly_prices['daily_price'] * hourly_multiplier
        
        # Add some hourly noise
        np.random.seed(42)
        hourly_prices['price_per_mwh'] += np.random.randn(len(hourly_prices)) * (hourly_prices['price_per_mwh'] * 0.05)
        
        # Clean up
        self.price_data = hourly_prices[['price_per_mwh']].dropna()
        
        print(f"Interpolated to {len(self.price_data)} hours")
        print(f"Hourly price range: ${self.price_data['price_per_mwh'].min():.2f} - ${self.price_data['price_per_mwh'].max():.2f}/MWh")
        print(f"Mean price: ${self.price_data['price_per_mwh'].mean():.2f}/MWh")
        
        # Save to cache
        print(f"Saving to cache: {cache_filename}")
        self.price_data.to_csv(cache_filename)
        
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
        
        # Initialize results
        results = {
            'hourly_data': pd.DataFrame(index=self.price_data.index),
            'revenues': {},
            'kpis': {}
        }
        
        # Add price and solar data
        results['hourly_data']['price_per_mwh'] = self.price_data['price_per_mwh']
        results['hourly_data']['solar_mw'] = solar_mw.reindex(self.price_data.index, fill_value=0)
        
        # Initialize battery variables
        battery_energy_mwh = 0
        results['hourly_data']['battery_charge_mw'] = 0.0
        results['hourly_data']['battery_discharge_mw'] = 0.0
        results['hourly_data']['battery_soc_mwh'] = 0.0
        results['hourly_data']['grid_sales_mw'] = 0.0
        results['hourly_data']['grid_purchases_mw'] = 0.0
        
        # Run arbitrage logic
        for i, (timestamp, row) in enumerate(results['hourly_data'].iterrows()):
            price = row['price_per_mwh']
            solar = row['solar_mw']
            
            # Calculate price thresholds using quantiles for better optimization
            day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(hours=24)
            day_prices = results['hourly_data'][(results['hourly_data'].index >= day_start) & 
                                              (results['hourly_data'].index < day_end)]['price_per_mwh']
            
            if len(day_prices) > 0:
                low_threshold = day_prices.quantile(0.33)  # Bottom 33% of prices
                high_threshold = day_prices.quantile(0.67)  # Top 33% of prices
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
            
            # Update results for this hour
            results['hourly_data'].loc[timestamp, 'battery_charge_mw'] = charge_mw
            results['hourly_data'].loc[timestamp, 'battery_discharge_mw'] = discharge_mw
            results['hourly_data'].loc[timestamp, 'battery_soc_mwh'] = battery_energy_mwh
            results['hourly_data'].loc[timestamp, 'grid_sales_mw'] = grid_sales
            results['hourly_data'].loc[timestamp, 'grid_purchases_mw'] = grid_purchases
        
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