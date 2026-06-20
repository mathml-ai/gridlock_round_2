import pandas as pd
import numpy as np
import h3
import json
import joblib
from datetime import datetime, timedelta

# =====================================================================
# 1. LOAD THE TRAINED AI MODEL
# =====================================================================
print("Loading V1 LightGBM Model...")
try:
    model = joblib.load('lgbm_model_v1.pkl')
except FileNotFoundError:
    print("Error: lgbm_model_v1.pkl not found. Make sure you saved it from the training step!")
    exit()

# The exact features the model expects
MODEL_FEATURES = [
    'hour', 'day_of_week', 'is_weekend', 
    'current_violations', 'heavy_vehicle_ratio', 'main_road_blocks',
    'count_last_1h', 'count_last_24h', 'growth_rate', 'neighbor_pressure'
]

# =====================================================================
# 2. DATA INGESTION (MOCK DB PULL)
# =====================================================================
def fetch_live_and_historical_data():
    """
    In production, this queries your PostgreSQL/MongoDB database.
    For the hackathon, I have simulated pulling the last 25 hours of data 
    from given dataset to satisfy the 24-hour lag requirement.
    """
    print("Fetching live data + 24-hour historical tail...")
    df_raw = pd.read_csv('sample.csv') # Replace with your actual data source
    
    # Optional Hackathon trick: Filter to just the last 25 hours of data here 
    # to simulate a live database query.
    return df_raw

# =====================================================================
# 3. BASE PREPROCESSING & SPATIAL INDEXING
# =====================================================================
def preprocess_raw_data(df):
    print("Sanitizing live feed and spatial indexing...")
    
    # 🚨 EDGE CASE HANDLING 1: Critical Missing Data 🚨
    # If we don't know WHERE or WHEN it happened, the data is useless. Drop it.
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime'])
    
    if df.empty:
        print("Warning: All live data was corrupted or missing coordinates.")
        return df

    # 🚨 EDGE CASE HANDLING 2: Non-Critical Missing Data 🚨
    # If vehicle type is blank, assume standard light vehicle (UNKNOWN)
    df['vehicle_type'] = df['vehicle_type'].fillna('UNKNOWN')
    
    # If violation type is blank, provide stringified empty JSON array
    df['violation_type'] = df['violation_type'].fillna('[]')

    # Timezone fix (UTC to IST)
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed', utc=True)
    df['created_datetime'] = df['created_datetime'].dt.tz_convert('Asia/Kolkata')
    
    df['date'] = df['created_datetime'].dt.date
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Parse JSON Violations & Vehicle Types
    def parse_violations(v):
        try: return json.loads(v)
        except: return []
        
    df['violation_list'] = df['violation_type'].apply(parse_violations)
    df['violation_str'] = df['violation_list'].astype(str).str.upper()
    df['is_main_road'] = df['violation_str'].apply(lambda x: 1 if 'MAIN ROAD' in x or 'CROSSING' in x else 0)
    
    heavy_vehicles = ['TRUCK', 'TANKER', 'BUS', 'MAXI-CAB']
    df['is_heavy'] = df['vehicle_type'].apply(lambda x: 1 if str(x).upper() in heavy_vehicles else 0)

    # H3 Spatial Snapping (V4 API)
    def get_h3_cell(lat, lon):
        if pd.isna(lat) or pd.isna(lon): return None
        return h3.latlng_to_cell(lat, lon, 9)
        
    df['h3_cell'] = df.apply(lambda row: get_h3_cell(row['latitude'], row['longitude']), axis=1)
    
    # Drop any rows where H3 snapping failed (due to corrupted extreme lat/lons)
    return df.dropna(subset=['h3_cell'])

# =====================================================================
# 4. DYNAMIC FEATURE ENGINEERING & LAG CALCULATION
# =====================================================================
def engineer_live_features(df):
    print("Aggregating Cell-Hours and calculating rolling momentum...")
    
    # Aggregate into cell-hour blocks
    grid_hourly = df.groupby(['h3_cell', 'date', 'hour', 'day_of_week', 'is_weekend']).agg(
        current_violations=('id', 'count'),
        heavy_vehicles_count=('is_heavy', 'sum'),
        main_road_blocks=('is_main_road', 'sum')
    ).reset_index()

    grid_hourly['heavy_vehicle_ratio'] = grid_hourly['heavy_vehicles_count'] / grid_hourly['current_violations']
    grid_hourly = grid_hourly.sort_values(by=['h3_cell', 'date', 'hour']).reset_index(drop=True)

    # Calculate Lags using the historical tail
    grid_hourly['count_last_1h'] = grid_hourly.groupby('h3_cell')['current_violations'].shift(1).fillna(0)
    grid_hourly['count_last_24h'] = grid_hourly.groupby('h3_cell')['current_violations'].shift(24).fillna(0)
    grid_hourly['growth_rate'] = grid_hourly['current_violations'] / (grid_hourly['count_last_1h'] + 1)

    # Calculate Spillover Risk (H3 V4 API)
    time_cell_dict = grid_hourly.set_index(['date', 'hour', 'h3_cell'])['current_violations'].to_dict()
    
    def calculate_neighbor_pressure(row):
        neighbors = h3.grid_disk(row['h3_cell'], 1)
        if row['h3_cell'] in neighbors:
            neighbors.remove(row['h3_cell'])
        
        pressure = 0
        for neighbor in neighbors:
            pressure += time_cell_dict.get((row['date'], row['hour'], neighbor), 0)
        return pressure

    grid_hourly['neighbor_pressure'] = grid_hourly.apply(calculate_neighbor_pressure, axis=1)
    
    return grid_hourly

# =====================================================================
# 5. ISOLATE LIVE HOUR & PREDICT
# =====================================================================
def generate_predictions(grid_hourly):
    print("Running LightGBM Inference on all active H3 cells...")

    # Get latest available state for every H3 cell
    live_state = (
        grid_hourly
        .sort_values(['h3_cell', 'date', 'hour'])
        .groupby('h3_cell')
        .tail(1)
        .copy()
    )

    if live_state.empty:
        print("No live data available for inference.")
        return None

    # Predict next-hour violations
    X_live = live_state[MODEL_FEATURES]

    live_state['predicted_next_hour'] = model.predict(X_live)

    # Round predictions
    live_state['predicted_next_hour'] = (
        live_state['predicted_next_hour']
        .clip(lower=0)
        .round()
        .astype(int)
    )

    return live_state[
        [
            'h3_cell',
            'date',
            'hour',
            'current_violations',
            'predicted_next_hour',
            'growth_rate',
            'main_road_blocks',
            'neighbor_pressure'
        ]
    ]

# =====================================================================
# 6. EXECUTION RUNNER
# =====================================================================
if __name__ == "__main__":

    print("\n--- STARTING LIVE PIPELINE ---")

    raw_data = fetch_live_and_historical_data()

    clean_data = preprocess_raw_data(raw_data)

    if clean_data.empty:
        print("No usable data found.")
        exit()

    feature_data = engineer_live_features(clean_data)

    final_predictions = generate_predictions(feature_data)

    if final_predictions is None:
        print("No predictions generated.")
        exit()

    print("\n🚨 TOP PREDICTED HOTSPOTS 🚨")

    final_predictions = final_predictions.sort_values(
        by='predicted_next_hour',
        ascending=False
    )

    print(
        final_predictions[
            [
                'h3_cell',
                'current_violations',
                'predicted_next_hour',
                'growth_rate',
                'neighbor_pressure'
            ]
        ].head(20)
    )

    # Attach map coordinates
    final_predictions['hex_lat'] = final_predictions['h3_cell'].apply(
        lambda x: h3.cell_to_latlng(x)[0]
    )

    final_predictions['hex_lon'] = final_predictions['h3_cell'].apply(
        lambda x: h3.cell_to_latlng(x)[1]
    )

    final_predictions.to_csv(
        'live_predictions.csv',
        index=False
    )

    print(
        f"\n✅ Generated predictions for "
        f"{len(final_predictions)} H3 cells."
    )

    print(
        "Saved: live_predictions.csv"
    )