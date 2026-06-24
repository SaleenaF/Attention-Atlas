"""
Attention Atlas Data Compiler
Scans the data folder and bundles all individual YouTube Studio exports 
into a single, multi-tab master Excel workbook.
"""

import os
import glob
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Step up out of src/
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "youtube_master_analytics.xlsx")

def clean_sheet_name(filename):
    """
    Cleans up the file name to create a valid Excel sheet tab name.
    Excel tabs have a strict 31-character limit.
    """
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # Simplify common prefixes to maximize character room
    name = name.replace("Chart data_", "Chart_")
    name = name.replace("Table data_", "Table_")
    name = name.replace("Totals_", "Totals_")
    
    # Return truncated name to respect Excel's 31-character constraint
    return name[:31]

def build_master_workbook():
    print("🚀 Initializing Attention Atlas Data Compiler...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Could not find the folder at '{DATA_DIR}'. Make sure your files are inside a 'data' folder.")
        return

    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("⚠️ No CSV files found in your data folder! Double check if they are saved as .csv or .xlsx.")
        return

    print(f"📦 Found {len(csv_files)} tracking files. Beginning compilation step...")
    
    # Open the single master Excel file writer
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for file_path in sorted(csv_files):
            file_name = os.path.basename(file_path)
            sheet_name = clean_sheet_name(file_name)
            
            try:
                # Read the CSV file (using error_bad_lines/on_bad_lines to skip header notes if YouTube included any)
                df = pd.read_csv(file_path, on_bad_lines='skip')
                
                # Write to its own tab inside the Excel workbook
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"   ✅ Bundled: '{file_name}' -> Tab: [{sheet_name}]")
                
            except Exception as e:
                print(f"   ❌ Skipped '{file_name}' due to error: {e}")
                
    print("\n🎉 Compilation Complete!")
    print(f"💾 Your master file is ready here: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_master_workbook()