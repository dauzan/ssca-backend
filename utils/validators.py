import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re


class DataValidator:
    """
    Main data validation class for SSCA
    """
    
    def __init__(self):
        self.validation_results = []
        self.error_log = []
        
    def validate_operational_logs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate operational logs data
        
        Required columns:
        - date, facility_id, production_units, kwh_consumed, 
          gas_m3, runtime_hours, emissions_kg_co2e
        
        Returns:
            {
                'valid': bool,
                'errors': list,
                'warnings': list,
                'quality_score': float (0-100),
                'completeness': float (0-100)
            }
        """
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = [
            'date', 'facility_id', 'production_units', 
            'kwh_consumed', 'gas_m3', 'runtime_hours', 
            'emissions_kg_co2e'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return {
                'valid': False,
                'errors': errors,
                'warnings': warnings,
                'quality_score': 0,
                'completeness': 0
            }
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                if pct > 10:
                    errors.append(f"{col} has {pct:.1f}% missing values (threshold: 10%)")
                elif pct > 5:
                    warnings.append(f"{col} has {pct:.1f}% missing values")
        
        # Check for negative values
        numeric_columns = ['production_units', 'kwh_consumed', 'gas_m3', 
                          'runtime_hours', 'emissions_kg_co2e']
        
        for col in numeric_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    errors.append(f"{col} has {negative_count} negative values")
        
        # Check for outliers using IQR method
        for col in numeric_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))]
                
                if len(outliers) > 0:
                    pct = (len(outliers) / len(df)) * 100
                    if pct > 5:
                        warnings.append(f"{col} has {len(outliers)} outliers ({pct:.1f}%)")
        
        # Check date format and continuity
        try:
            df['date'] = pd.to_datetime(df['date'])
            date_gaps = df['date'].diff().dt.days
            large_gaps = date_gaps[date_gaps > 7].count()
            if large_gaps > 0:
                warnings.append(f"Found {large_gaps} date gaps larger than 7 days")
        except Exception as e:
            errors.append(f"Date format error: {str(e)}")
        
        # Check runtime hours (should be <= 24)
        if 'runtime_hours' in df.columns:
            invalid_hours = df[df['runtime_hours'] > 24]
            if len(invalid_hours) > 0:
                errors.append(f"Found {len(invalid_hours)} records with runtime_hours > 24")
        
        # Calculate quality score
        completeness = 100 - (df[required_columns].isnull().sum().sum() / 
                             (len(df) * len(required_columns)) * 100)
        
        error_penalty = len(errors) * 10
        warning_penalty = len(warnings) * 5
        quality_score = max(0, 100 - error_penalty - warning_penalty)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': round(quality_score, 2),
            'completeness': round(completeness, 2),
            'row_count': len(df),
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            }
        }
    
    def validate_logistics_routes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate logistics routes data
        
        Required columns:
        - shipment_id, shipment_date, origin, destination, 
          transport_mode, distance_km, cargo_weight_tons
        """
        errors = []
        warnings = []
        
        required_columns = [
            'shipment_id', 'shipment_date', 'origin', 'destination',
            'transport_mode', 'distance_km', 'cargo_weight_tons'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check transport modes are valid
        valid_modes = ['truck', 'ship', 'rail', 'air', 'barge']
        if 'transport_mode' in df.columns:
            invalid_modes = df[~df['transport_mode'].isin(valid_modes)]
            if len(invalid_modes) > 0:
                errors.append(f"Found {len(invalid_modes)} records with invalid transport mode")
        
        # Check distance is reasonable
        if 'distance_km' in df.columns:
            # Maximum reasonable distance (earth circumference)
            max_distance = 40075  # km
            invalid_distance = df[(df['distance_km'] <= 0) | (df['distance_km'] > max_distance)]
            if len(invalid_distance) > 0:
                errors.append(f"Found {len(invalid_distance)} records with invalid distance")
        
        # Check cargo weight
        if 'cargo_weight_tons' in df.columns:
            invalid_weight = df[df['cargo_weight_tons'] <= 0]
            if len(invalid_weight) > 0:
                errors.append(f"Found {len(invalid_weight)} records with invalid weight")
            
            # Check for unrealistic weights (e.g., > 10000 tons for trucks)
            if 'transport_mode' in df.columns:
                heavy_trucks = df[(df['transport_mode'] == 'truck') & 
                                 (df['cargo_weight_tons'] > 100)]
                if len(heavy_trucks) > 0:
                    warnings.append(f"Found {len(heavy_trucks)} truck shipments > 100 tons")
        
        # Check date format
        try:
            df['shipment_date'] = pd.to_datetime(df['shipment_date'])
        except Exception as e:
            errors.append(f"Date format error: {str(e)}")
        
        # Calculate completeness
        completeness = 100 - (df[required_columns].isnull().sum().sum() / 
                             (len(df) * len(required_columns)) * 100)
        
        quality_score = max(0, 100 - len(errors) * 10 - len(warnings) * 5)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': round(quality_score, 2),
            'completeness': round(completeness, 2),
            'row_count': len(df)
        }
    
    def validate_supplier_esg(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate supplier ESG data
        
        Required columns:
        - supplier_id, supplier_name, country, tier,
          scope1_emissions_kg, scope2_emissions_kg, scope3_emissions_kg
        """
        errors = []
        warnings = []
        
        required_columns = [
            'supplier_id', 'supplier_name', 'country', 'tier',
            'scope1_emissions_kg', 'scope2_emissions_kg'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check tier values
        valid_tiers = ['Tier-1', 'Tier-2', 'Tier-3']
        if 'tier' in df.columns:
            invalid_tiers = df[~df['tier'].isin(valid_tiers)]
            if len(invalid_tiers) > 0:
                errors.append(f"Found {len(invalid_tiers)} records with invalid tier")
        
        # Check emissions are non-negative
        emission_columns = ['scope1_emissions_kg', 'scope2_emissions_kg', 'scope3_emissions_kg']
        for col in emission_columns:
            if col in df.columns:
                negative = df[df[col] < 0]
                if len(negative) > 0:
                    errors.append(f"{col} has {len(negative)} negative values")
        
        # Check total emissions consistency
        if all(col in df.columns for col in ['scope1_emissions_kg', 'scope2_emissions_kg', 
                                              'scope3_emissions_kg', 'total_emissions_kg']):
            df['calculated_total'] = (df['scope1_emissions_kg'] + 
                                     df['scope2_emissions_kg'] + 
                                     df['scope3_emissions_kg'])
            
            # Allow 1% tolerance for rounding
            mismatch = df[abs(df['calculated_total'] - df['total_emissions_kg']) > 
                         (df['total_emissions_kg'] * 0.01)]
            
            if len(mismatch) > 0:
                warnings.append(f"Found {len(mismatch)} records where Scope 1+2+3 != Total")
        
        # Check renewable energy percentage
        if 'renewable_energy_pct' in df.columns:
            invalid_pct = df[(df['renewable_energy_pct'] < 0) | 
                           (df['renewable_energy_pct'] > 100)]
            if len(invalid_pct) > 0:
                errors.append(f"Found {len(invalid_pct)} records with invalid renewable %")
        
        # Check data quality score
        if 'data_quality_score' in df.columns:
            low_quality = df[df['data_quality_score'] < 0.5]
            if len(low_quality) > 0:
                warnings.append(f"Found {len(low_quality)} suppliers with low data quality (<0.5)")
        
        # Calculate completeness
        completeness = 100 - (df[required_columns].isnull().sum().sum() / 
                             (len(df) * len(required_columns)) * 100)
        
        quality_score = max(0, 100 - len(errors) * 10 - len(warnings) * 5)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_score': round(quality_score, 2),
            'completeness': round(completeness, 2),
            'row_count': len(df),
            'tier_distribution': df['tier'].value_counts().to_dict() if 'tier' in df.columns else {}
        }
    
    def validate_forecast_request(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate emission forecast API request
        """
        errors = []
        
        if 'facility_id' not in data:
            errors.append("Missing required field: facility_id")
        
        if 'horizon_days' in data:
            horizon = data['horizon_days']
            if not isinstance(horizon, int):
                errors.append("horizon_days must be an integer")
            elif horizon < 1 or horizon > 90:
                errors.append("horizon_days must be between 1 and 90")
        
        return len(errors) == 0, errors
    
    def validate_supplier_risk_request(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate supplier risk scoring API request
        """
        errors = []
        
        if 'supplier_data' not in data:
            errors.append("Missing required field: supplier_data")
            return False, errors
        
        supplier_data = data['supplier_data']
        
        # Check required fields
        if 'emissions' in supplier_data and supplier_data['emissions'] < 0:
            errors.append("emissions must be non-negative")
        
        if 'renewable_pct' in supplier_data:
            pct = supplier_data['renewable_pct']
            if pct < 0 or pct > 100:
                errors.append("renewable_pct must be between 0 and 100")
        
        return len(errors) == 0, errors
    
    def validate_optimization_request(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate optimization scenario API request
        """
        errors = []
        
        if 'scenario' not in data:
            errors.append("Missing required field: scenario")
            return False, errors
        
        scenario = data['scenario']
        
        if 'modal_shift_pct' in scenario:
            pct = scenario['modal_shift_pct']
            if not isinstance(pct, (int, float)) or pct < 0 or pct > 100:
                errors.append("modal_shift_pct must be between 0 and 100")
        
        if 'renewable_increase_pct' in scenario:
            pct = scenario['renewable_increase_pct']
            if not isinstance(pct, (int, float)) or pct < 0 or pct > 100:
                errors.append("renewable_increase_pct must be between 0 and 100")
        
        return len(errors) == 0, errors
    
    def check_outliers_iqr(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Returns: Boolean series where True indicates outlier
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_country_code(self, code: str) -> bool:
        """Validate ISO country code (2 letters)"""
        return len(code) == 2 and code.isalpha() and code.isupper()
    
    def sanitize_text_input(self, text: str, max_length: int = 1000) -> str:
        """
        Sanitize text input to prevent injection attacks
        """
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\';]', '', text)
        # Limit length
        text = text[:max_length]
        return text.strip()

def validate_data(data, data_type: str) -> Dict[str, Any]:
    """
    Convenience function to validate data
    
    Args:
        data: DataFrame or dict to validate
        data_type: 'operational', 'logistics', 'supplier', 'forecast_request', etc.
    
    Returns:
        Validation result dictionary
    """
    validator = DataValidator()
    
    if data_type == 'operational':
        return validator.validate_operational_logs(data)
    elif data_type == 'logistics':
        return validator.validate_logistics_routes(data)
    elif data_type == 'supplier':
        return validator.validate_supplier_esg(data)
    elif data_type == 'forecast_request':
        is_valid, errors = validator.validate_forecast_request(data)
        return {'valid': is_valid, 'errors': errors}
    elif data_type == 'risk_request':
        is_valid, errors = validator.validate_supplier_risk_request(data)
        return {'valid': is_valid, 'errors': errors}
    elif data_type == 'optimization_request':
        is_valid, errors = validator.validate_optimization_request(data)
        return {'valid': is_valid, 'errors': errors}
    else:
        return {'valid': False, 'errors': [f'Unknown data type: {data_type}']}