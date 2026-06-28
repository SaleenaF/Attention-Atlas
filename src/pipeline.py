import pandas as pd
import sqlite3
import numpy as np
import glob
import os

def run_process():
    # Looking exactly where you specified
    target_dir = r"C:\Users\salee\Attention-Atlas\data"
    
    print(f"DEBUG: Manually checking path: {target_dir}")
    
    if not os.path.exists(target_dir):
        print(f"❌ ERROR: The folder {target_dir} does not exist!")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    print(f"DEBUG: Found {len(csv_files)} CSV files.")

    # Databases in your current root folder (C:\Users\salee\Attention-Atlas)
    master_conn = sqlite3.connect("attention_atlas.db")
    anon_conn = sqlite3.connect("anonymized_attention_atlas.db")
    
    for file in csv_files:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        
        # Mapping based on your files
        if 'Content' in df.columns and 'Video title' in df.columns:
            table_name = 'videos'
        elif 'Geography' in df.columns:
            table_name = 'geography'
        elif 'Cities' in df.columns or 'City name' in df.columns:
            table_name = 'cities'
        else:
            table_name = filename.replace(".csv", "").lower().replace(" ", "_")
        
        # Load to Master
        df.to_sql(table_name, master_conn, if_exists="replace", index=False)
        
        # Anonymize
        if "Video title" in df.columns: df["Video title"] = "Anonymized"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = (df[col] * np.random.uniform(0.95, 1.05, len(df))).astype(int if df[col].dtype == 'int64' else float)
            
        # Load to Anonymized
        df.to_sql(table_name, anon_conn, if_exists="replace", index=False)
        print(f"✅ Processed: {filename} -> Table: {table_name}")

    master_conn.close()
    anon_conn.close()
    print("Done.")

if __name__ == "__main__":
    run_process()