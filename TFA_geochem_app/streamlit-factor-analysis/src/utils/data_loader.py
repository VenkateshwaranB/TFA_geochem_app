import pandas as pd

def load_excel_data(file_path, sheet_name=None):
    """Load data from an Excel file."""
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def load_sample_data():
    """Load sample data from the predefined Excel file."""
    return load_excel_data('data/sample_data.xlsx')