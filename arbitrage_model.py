import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# class SolarStorageArbitrage:
#     """
#     A comprehensive model for calculating the economic value of adding battery storage
#     to a solar project through energy arbitrage.
    
#     This model demonstrates how battery storage can increase revenue by storing cheap
#     midday energy and selling it during expensive evening peaks.
#     """
    
#     def __init__(self):
#         # Initialize instance variables for data storage
#         # These will be populated during simulation runs
#         self.price_data = None  # Hourly electricity price data (DataFrame)
#         self.solar_profile = None  # Hourly solar generation profile (Series)
#         self.results = None  # Simulation results dictionary
        
#     def load_price_data(self, source_name="California (CAISO)"):
#         """
#         Load electricity price data from various sources.
        
#         Args:
#             source_name (str): Name of the price market to load
            
#         Returns:
#             pd.DataFrame: Clean price data with datetime index
#         """
#         # INPUT VALIDATION: Check for valid market selection
#         if source_name == "California (CAISO)":
#             # For demo purposes, create synthetic CAISO-like data
#             # In production, this would pull from EIA API or CAISO CSV
#             return self._create_synthetic_caiso_data()
#         elif source_name == "Texas (ERCOT)":
#             # For demo purposes, create synthetic ERCOT-like data
#             return self._create_synthetic_ercot_data()
#         else:
#             # ERROR HANDLING: Raise exception for unknown markets
#             raise ValueError(f"Unknown price source: {source_name}")
    
#     def _create_synthetic_caiso_data(self):
#         """Create synthetic CAISO-like price data with duck curve characteristics."""
#         # DATA GENERATION: Create a full year of hourly data (8,760 hours)
#         # This simulates real electricity market data with realistic patterns
#         start_date = datetime(2024, 1, 1)
#         end_date = datetime(2024, 12, 31, 23, 0)
#         date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
#         # PRICE MODELING: Create realistic electricity price patterns
#         # Base price with seasonal variation (higher in summer/winter)
#         seasonal_base = 50 + 20 * np.sin(2 * np.pi * np.arange(len(date_range)) / (365.25 * 24))
        
#         # Daily duck curve pattern (high morning/evening, low midday)
#         # This simulates the "duck curve" where solar generation depresses midday prices
#         hour_of_day = date_range.hour
#         daily_pattern = 30 + 40 * np.sin(np.pi * (hour_of_day - 6) / 12) + \
#                        20 * np.sin(np.pi * (hour_of_day - 18) / 6)
        
#         # Add realistic market volatility
#         noise = np.random.normal(0, 5, len(date_range))
        
#         # Combine all effects
#         prices = seasonal_base + daily_pattern + noise
        
#         # Ensure positive prices
#         prices = np.maximum(prices, 10)
        
#         # Create some extreme price events (like heat waves)
#         # BUG FIX: Convert prices to numpy array to avoid pandas indexing issues
#         prices_array = np.array(prices)
#         extreme_days = np.random.choice(len(date_range), size=20, replace=False)
#         for day in extreme_days:
#             if 14 <= date_range[day].hour <= 20:  # Peak hours
#                 prices_array[day] *= np.random.uniform(2, 4)
#         prices = prices_array
        
#         # Create DataFrame with proper indexing
#         df = pd.DataFrame({
#             'timestamp': date_range,
#             'price_per_mwh': prices
#         })
#         return df.set_index('timestamp')
    
#     def _create_synthetic_ercot_data(self):
#         """Create synthetic ERCOT-like price data with high volatility."""
#         # Create a full year of hourly data
#         start_date = datetime(2024, 1, 1)
#         end_date = datetime(2024, 12, 31, 23, 0)
#         date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
#         # Base price with seasonal variation
#         seasonal_base = 45 + 25 * np.sin(2 * np.pi * np.arange(len(date_range)) / (365.25 * 24))
        
#         # Daily pattern with higher volatility
#         hour_of_day = date_range.hour
#         daily_pattern = 25 + 30 * np.sin(np.pi * (hour_of_day - 6) / 12) + \
#                        15 * np.sin(np.pi * (hour_of_day - 18) / 6)
        
#         # Higher volatility for ERCOT
#         noise = np.random.normal(0, 8, len(date_range))
        
#         # Combine all effects
#         prices = seasonal_base + daily_pattern + noise
        
#         # Ensure positive prices
#         prices = np.maximum(prices, 5)
        
#         # Create more extreme price events (ERCOT is known for volatility)
#         # BUG FIX: Convert prices to numpy array to avoid pandas indexing issues
#         prices_array = np.array(prices)
#         extreme_days = np.random.choice(len(date_range), size=50, replace=False)
#         for day in extreme_days:
#             if 14 <= date_range[day].hour <= 20:  # Peak hours
#                 prices_array[day] *= np.random.uniform(3, 8)
#         prices = prices_array
        
#         # Create DataFrame with proper indexing
#         df = pd.DataFrame({
#             'timestamp': date_range,
#             'price_per_mwh': prices
#         })
#         return df.set_index('timestamp')
    
#     def generate_solar_profile(self, peak_mw, num_days=None, price_data=None):
#         """
#         Generate synthetic solar generation profile.
        
#         Args:
#             peak_mw (float): Peak solar capacity in MW
#             num_days (int): Number of days to generate (default: match price data)
#             price_data (pd.DataFrame): Price data to match time range
            
#         Returns:
#             pd.Series: Hourly solar generation in kW
#         """
#         # BUG FIX: Use passed price_data or fall back to instance variable
#         if price_data is not None:
#             reference_data = price_data
#         elif self.price_data is not None:
#             reference_data = self.price_data
#         else:
#             # FALLBACK: Create default time range if no reference data
#             start_date = datetime(2024, 1, 1)
#             end_date = datetime(2024, 12, 31, 23, 0)
#             reference_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='h'))
        
#         if num_days is None:
#             num_days = len(reference_data) // 24
        
#         # Create hourly index for the period
#         start_date = reference_data.index[0]
#         end_date = start_date + timedelta(days=num_days)
#         hourly_index = pd.date_range(start=start_date, end=end_date, freq='h')
        
#         # Generate solar profile using sine wave
#         hour_of_day = hourly_index.hour
#         solar_profile = []
        
#         for hour in hour_of_day:
#             if 6 <= hour <= 18:  # Daylight hours
#                 # Sine wave that peaks at noon
#                 output_kw = max(0, np.sin(np.pi * (hour - 6) / 12)) * peak_mw * 1000
#             else:
#                 output_kw = 0
            
#             solar_profile.append(output_kw)
        
#         return pd.Series(solar_profile, index=hourly_index)

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

class SolarStorageArbitrage:
    """
    A comprehensive model for calculating the economic value of adding battery storage
    to a solar project using real-world electricity price data.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the model. Requires an EIA API key.
        
        Args:
            api_key (str): Your personal API key from the EIA.
        """
        # API Key Management
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            raise ValueError("EIA API key not provided. Pass it to the constructor or set the EIA_API_KEY environment variable.")
        
        # Instance variables for data storage
        self.price_data = None
        self.solar_profile = None
        self.results = None

    def _fetch_eia_data(self, series_id, start_date, end_date):
        """
        Fetches hourly data from the EIA API for a given series ID.
        
        Args:
            series_id (str): The specific data series ID from the EIA.
            start_date (datetime): The start date for the data pull.
            end_date (datetime): The end date for the data pull.
            
        Returns:
            pd.DataFrame: A clean DataFrame with a datetime index and price data.
        """
        # API v2 Endpoint for electricity data
        url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        
        # Format dates for the API request
        start_str = start_date.strftime('%Y-%m-%dT%H')
        end_str = end_date.strftime('%Y-%m-%dT%H')
        
        # Set up the API request parameters
        params = {
            'api_key': self.api_key,
            'frequency': 'hourly',
            'data[0]': 'value',
            'facets[seriesId][]': series_id,
            'start': start_str,
            'end': end_str,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': 0,
            'length': 5000  # Max length per request
        }
        
        print(f"Fetching data for {series_id} from {start_date} to {end_date}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
            
            # Parse the JSON response
            json_data = response.json()
            data = json_data.get('response', {}).get('data', [])
            
            if not data:
                print("Warning: No data returned from EIA API for the specified period.")
                return pd.DataFrame()

            # Convert to a Pandas DataFrame
            df = pd.DataFrame(data)
            
            # --- Data Cleaning and Formatting ---
            # 1. Rename columns for clarity
            df = df.rename(columns={'period': 'timestamp', 'value': 'price_per_mwh'})
            
            # 2. Convert timestamp to datetime objects (UTC) and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
            
            # 3. Convert price to a numeric type, handling errors
            df['price_per_mwh'] = pd.to_numeric(df['price_per_mwh'], errors='coerce')
            
            # 4. Handle any missing or null values
            df = df.dropna()
            
            print("Successfully fetched and processed real-world price data.")
            return df[['price_per_mwh']]

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from EIA API: {e}")
            return pd.DataFrame()
            
    def load_price_data(self, source_name="California (CAISO)", year=2023):
        """
        Load electricity price data from the EIA API.
        
        Args:
            source_name (str): Name of the price market to load.
            year (int): The full year of data to pull.
            
        Returns:
            pd.DataFrame: Clean price data with datetime index.
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59)
        
        # EIA Series IDs for Real-Time Hourly Locational Marginal Price (LMP)
        series_map = {
            "California (CAISO)": "ELEC.PRICE.CAISO-APND.RTLMP.H",
            "Texas (ERCOT)": "ELEC.PRICE.ERCOT-ALL.RTLMP.H",
            # Add other markets here as needed (e.g., PJM, MISO)
        }
        
        if source_name not in series_map:
            raise ValueError(f"Unknown price source: {source_name}. Available sources: {list(series_map.keys())}")
            
        series_id = series_map[source_name]
        
        self.price_data = self._fetch_eia_data(series_id, start_date, end_date)
        return self.price_data
        
    def generate_solar_profile(self, peak_mw):
        """Generates a solar profile that matches the loaded price data's index."""
        if self.price_data is None or self.price_data.empty:
            raise ValueError("Price data must be loaded before generating a solar profile.")
        
        hourly_index = self.price_data.index
        hour_of_day = hourly_index.hour
        
        # Generate solar profile using a sine wave based on the hour
        solar_profile = []
        for hour in hour_of_day:
            if 6 <= hour <= 18:  # Simple daylight hours
                output_kw = max(0, np.sin(np.pi * (hour - 6) / 12)) * peak_mw * 1000
            else:
                output_kw = 0
            solar_profile.append(output_kw)
        
        self.solar_profile = pd.Series(solar_profile, index=hourly_index)
        return self.solar_profile
    
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
        
        # CORE SIMULATION LOOP: Process each hour of the year
        # This is the heart of the arbitrage algorithm
        for i, (timestamp, row) in enumerate(price_data.iterrows()):
            # INPUT DATA: Get current hour's inputs
            # BUG PREVENTION: Check bounds to avoid index errors
            solar_gen_kw = solar_profile.iloc[i] if i < len(solar_profile) else 0
            market_price_per_mwh = row['price_per_mwh']
            market_price_per_kwh = market_price_per_mwh / 1000  # Convert to $/kWh
            
            # DISPATCH DECISION: Use daily average as threshold for charge/discharge
            # This is a simple but effective arbitrage strategy
            date_key = timestamp.date()
            daily_avg_price = daily_avg_prices.get(date_key, market_price_per_mwh)
            
            # BASELINE CALCULATION: Revenue from standalone solar
            standalone_revenue = solar_gen_kw * market_price_per_kwh
            hourly_revenues_standalone.append(standalone_revenue)
            
            # ARBITRAGE ALGORITHM: Core storage dispatch logic
            if market_price_per_mwh < daily_avg_price:
                # CHARGE MODE: Store energy when prices are low (midday)
                # This captures the arbitrage opportunity
                energy_available_to_charge = solar_gen_kw
                # CONSTRAINTS: Respect power limits, capacity limits, and available energy
                energy_to_charge = min(
                    energy_available_to_charge,  # Available solar generation
                    battery_power_kw,  # Battery power rating
                    battery_capacity_kwh - current_charge_kwh  # Remaining capacity
                )
                
                # BATTERY STATE UPDATE: Apply efficiency loss on charge
                # Using square root splits efficiency loss between charge and discharge
                current_charge_kwh += energy_to_charge * round_trip_efficiency_sqrt
                
                # GRID EXPORTS: Sell remaining solar energy to grid
                energy_sold_to_grid = solar_gen_kw - energy_to_charge
                energy_sold_solar.append(energy_sold_to_grid)
                energy_sold_battery.append(0)  # No battery discharge in charge mode
                
            else:
                # DISCHARGE MODE: Sell stored energy when prices are high (evening)
                # This is where the arbitrage value is captured
                energy_to_discharge = min(battery_power_kw, current_charge_kwh)
                
                # BATTERY STATE UPDATE: Reduce stored energy
                current_charge_kwh -= energy_to_discharge
                
                # GRID EXPORTS: Sell solar + battery energy to grid
                # Apply efficiency loss on discharge (other half of round-trip loss)
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
