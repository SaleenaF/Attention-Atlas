"""
Attention Atlas - Advanced Data Pipeline (Anonymized)
Processes your multi-tab 'youtube_master_analytics.xlsx' spreadsheet,
standardizes fields based on creative orientation, anonymizes creator identity,
and populates SQLite.
"""

import os
import pandas as pd
import sqlite3
import numpy as np

# Adjusts perfectly if run from inside the 'src/' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_EXCEL = os.path.join(BASE_DIR, "youtube_master_analytics.xlsx")
DB_PATH = os.path.join(BASE_DIR, "attention_atlas.db")

def run_advanced_pipeline():
    print("🚀 Booting Attention Atlas Advanced Data Pipeline...")
    
    if not os.path.exists(MASTER_EXCEL):
        print(f"❌ Error: Master file '{MASTER_EXCEL}' not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Process Core Video Content Performance Metrics
    print("📦 Processing core video catalog statistics...")
    df_content = pd.read_excel(MASTER_EXCEL, sheet_name="Table_data_content")
    df_content = df_content[df_content['Content'] != 'Total'].copy()
    
    df_content.columns = (
        df_content.columns.str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("%", "pct")
        .str.replace("-", "_")
    )
    
    df_content['publish_date'] = pd.to_datetime(df_content['video_publish_time'], errors='coerce')
    df_content['publish_year'] = df_content['publish_date'].dt.year
    df_content['publish_month'] = df_content['publish_date'].dt.month
    df_content['publish_dow'] = df_content['publish_date'].dt.strftime('%A')
    
    # Custom format assignment tracking true cinematic landscape vs vertical framing
    def determine_true_format(row):
        title = str(row['video_title']).lower()
        if 'animatic' in title or 'alhaitham' in title:
            return 'Long-form'
        elif '#shorts' in title or 'shorts' in title or 'clip' in title:
            return 'Shorts'
        return 'Shorts' if row['duration'] <= 60 else 'Long-form'
        
    df_content['format'] = df_content.apply(determine_true_format, axis=1)
    
    def assign_category(title):
        t = str(title).lower()
        if any(k in t for k in ['pull', 'wish', 'varka', 'gacha', 'pity', 'luck']):
            return 'Gacha / Pulls'
        elif any(k in t for k in ['animatic', 'art', 'draw', 'speedpaint', 'tutorial', 'sketch']):
            return 'Art / Creative'
        else:
            return 'Other Gaming / Clips'
            
    df_content['category'] = df_content['video_title'].apply(assign_category)

    # --- DATA ANONYMIZATION STEP ---
    print("🔒 Scrubbing identifying video titles and string identifiers...")
    
    # Sort by publish date so video numbers sequence chronologically (optional but cleaner)
    df_content = df_content.sort_values('publish_date').reset_index(drop=True)
    
    # Extract unique real IDs to map them consistently
    unique_ids = df_content['content'].unique()
    
    # Generate token maps
    id_mapping = {real_id: f"VID_{i+1:03d}" for i, real_id in enumerate(unique_ids)}
    title_mapping = {real_id: f"Content Project {i+1:03d}" for i, real_id in enumerate(unique_ids)}
    
    # Overwrite identifying columns entirely
    df_content['video_title'] = df_content['content'].map(title_mapping)
    df_content['content'] = df_content['content'].map(id_mapping)
    # --------------------------------

    df_content.to_sql("videos", conn, if_exists="replace", index=False)

    # 2. Process Geography
    print("🗺️ Processing geographic data vectors...")
    df_geo = pd.read_excel(MASTER_EXCEL, sheet_name="Table_geography")
    df_geo = df_geo[df_geo['Geography'] != 'Total'].copy()
    
    iso_map = {
        "US": "USA", "CA": "CAN", "PH": "PHL", "ID": "IDN", "GB": "GBR",
        "IN": "IND", "DE": "DEU", "RU": "RUS", "MY": "MYS", "BR": "BRA",
        "AU": "AUS", "PL": "POL", "FR": "FRA", "VN": "VNM", "KZ": "KAZ",
        "MA": "MAR", "LT": "LTU", "NP": "NPL", "DZ": "DZA", "HR": "HRV",
        "KH": "KHM", "TT": "TTO", "EC": "ECU", "JM": "JAM", "TW": "TWN",
        "BN": "BRN", "BY": "BLR", "CR": "CRI", "BO": "BOL", "VE": "VEN", "MN": "MNG"
    }
    df_geo['Geography'] = df_geo['Geography'].map(iso_map)
    df_geo = df_geo.dropna(subset=['Geography'])
    df_geo.columns = ['geography', 'views', 'watch_time_hours', 'average_view_duration']
    df_geo.to_sql("geography", conn, if_exists="replace", index=False)
    
    # 2.5 Process Cities (Bubble Map)
    print("🏙️ Processing demographic city clusters...")
    if "Table_cities" in pd.ExcelFile(MASTER_EXCEL).sheet_names:
        df_cities = pd.read_excel(MASTER_EXCEL, sheet_name="Table_cities")
        df_cities = df_cities[df_cities['Cities'] != 'Total'].copy()
        
        # Clean column names to lowercase and standard underscore formats
        df_cities.columns = (
            df_cities.columns.str.lower()
            .str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
        )
        
        # Split the city_name or clean it up if necessary 
        # (Plotly's scatter_geo works best if we pass the string cleanly)
        df_cities = df_cities.dropna(subset=['city_name'])
        
        df_cities.to_sql("cities", conn, if_exists="replace", index=False)
        print("   ✅ Loaded 'cities' table successfully.")
    else:
        print("   ⚠️ Warning: 'Table_cities' sheet not found in master workbook.")

    # 3. Process Daily Channel Aggregations
    print("📈 Processing historical daily timeline streams...")
    df_date = pd.read_excel(MASTER_EXCEL, sheet_name="Table_date")
    df_date = df_date[df_date['Date'] != 'Total'].copy()
    df_date.columns = df_date.columns.str.lower().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
    df_date['date'] = pd.to_datetime(df_date['date'], errors='coerce')
    df_date.to_sql("channel_daily_totals", conn, if_exists="replace", index=False)
    
    conn.close()
    print("\n🎉 Database rebuild complete! Real identifiers have been completely stripped.")

if __name__ == "__main__":
    run_advanced_pipeline()