import requests
import json

# Test script to discover what's actually available in EIA API v2
API_KEY = "pWeHccHjT5QVlxMMoKpKmAKJLsqAGZZM9vdDREYS"

def test_endpoint(path):
    """Query API metadata to see what's available"""
    url = f"https://api.eia.gov/v2/{path}"
    params = {'api_key': API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n{'='*80}")
        print(f"ENDPOINT: {path}")
        print(f"{'='*80}")
        
        # Print response structure
        if 'response' in data:
            resp = data['response']
            
            # Check for available data types
            if 'data' in resp:
                print(f"\nData fields: {list(resp['data'].keys())}")
                for field, info in resp['data'].items():
                    print(f"  {field}: {info}")
            
            # Check for facets
            if 'facets' in resp:
                print(f"\nFacets:")
                for facet in resp['facets']:
                    print(f"  {facet}")
            
            # Check for frequency options
            if 'frequency' in resp:
                print(f"\nFrequency options: {resp['frequency']}")
                
        print(json.dumps(data, indent=2)[:1000])  # Print first 1000 chars
        
    except Exception as e:
        print(f"ERROR testing {path}: {e}")

# Test key endpoints
print("Testing EIA API v2 endpoints for electricity/rto data...")

test_endpoint("electricity/rto")  # See what's under RTO
test_endpoint("electricity/rto/daily-region-data")  # Daily regional data
test_endpoint("electricity/rto/region-data")  # Hourly regional data

# Try to get facet info for daily-region-data
print("\n\nTesting facets for daily-region-data...")
test_endpoint("electricity/rto/daily-region-data/facet/type")