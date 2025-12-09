import pandas as pd
import numpy as np
import json
from pathlib import Path

class DataProcessor:
    """
    Handle data loading, validation, and preprocessing
    """
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
    
    def load_suppliers(self):
        """Load supplier directory"""
        # Check if file exists, otherwise return demo data
        supplier_file = self.data_dir / 'sample_supplier_esg.csv'
        
        if supplier_file.exists():
            df = pd.read_csv(supplier_file)
            return df.to_dict('records')
        else:
            # Return demo data
            return [
                {
                    'id': 'S-001',
                    'name': 'Alpha Steel',
                    'country': 'CN',
                    'emissions': 12000,
                    'risk': 88,
                    'tier': 'Tier-1',
                    'certifications': ['ISO14001'],
                    'renewable_pct': 15
                },
                {
                    'id': 'S-002',
                    'name': 'Beta Logistics',
                    'country': 'ID',
                    'emissions': 5200,
                    'risk': 72,
                    'tier': 'Tier-1',
                    'certifications': ['ISO14001', 'CDP'],
                    'renewable_pct': 25
                },
                {
                    'id': 'S-003',
                    'name': 'Gamma Parts',
                    'country': 'VN',
                    'emissions': 2300,
                    'risk': 45,
                    'tier': 'Tier-2',
                    'certifications': ['ISO14001'],
                    'renewable_pct': 40
                },
                {
                    'id': 'S-004',
                    'name': 'Delta Energy',
                    'country': 'US',
                    'emissions': 54000,
                    'risk': 95,
                    'tier': 'Tier-1',
                    'certifications': [],
                    'renewable_pct': 8
                }
            ]
    
    def load_operational_logs(self):
        """Load operational logs for forecasting"""
        log_file = self.data_dir / 'sample_operational_logs.csv'
        
        if log_file.exists():
            return pd.read_csv(log_file)
        else:
            # Generate synthetic data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            return pd.DataFrame({
                'date': dates,
                'facility_id': 'F001',
                'production_units': np.random.randint(800, 1200, len(dates)),
                'kwh_consumed': np.random.randint(15000, 25000, len(dates)),
                'emissions_kg_co2e': np.random.randint(45000, 65000, len(dates))
            })
    
    def validate_data_quality(self, df, required_columns):
        """Check data completeness and quality"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            return {'valid': False, 'missing_columns': missing_cols}
        
        null_pct = df.isnull().sum() / len(df) * 100
        
        return {
            'valid': True,
            'completeness': 100 - null_pct.mean(),
            'row_count': len(df)
        }