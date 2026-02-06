from fastapi import Form
import re
import csv
import json
from datetime import datetime
import pandas as pd
import os
from src.config import LEADS_FILE


def save_lead_to_excel(lead_data):
    """
    Appends a dictionary of lead data to an Excel file.
    """
    # Add a timestamp to the data
    lead_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a DataFrame for the new single row
    new_row_df = pd.DataFrame([lead_data])
    
    if not os.path.exists(LEADS_FILE):
        # File doesn't exist: Create it with headers
        new_row_df.to_excel(LEADS_FILE, index=False, engine='openpyxl')
    else:
        # File exists: Append to it
        # We read existing data first (safest way for Excel)
        existing_df = pd.read_excel(LEADS_FILE)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        updated_df.to_excel(LEADS_FILE, index=False, engine='openpyxl')
        
    print(f"💾 Saved lead to {LEADS_FILE}")
    
    