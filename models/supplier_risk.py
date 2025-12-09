import mindspore as ms
import mindspore.nn as nn
import numpy as np

class SupplierRiskModel(nn.Cell):
    """
    Multi-dimensional risk scoring model
    Weights: Environmental(40%), Compliance(30%), Operational(20%), Reputational(10%)
    """
    def __init__(self, input_features=15, hidden_dim=64):
        super(SupplierRiskModel, self).__init__()
        
        self.fc1 = nn.Dense(input_features, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, 32)
        self.fc3 = nn.Dense(32, 1)  # Single risk score output
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        risk_score = self.sigmoid(self.fc3(x))
        return risk_score * 100  # Scale to 0-100


class SupplierRiskScorer:
    """
    Calculate composite risk score for suppliers
    """
    def __init__(self):
        self.model = SupplierRiskModel()
        self.is_loaded = True  # Demo mode
        
        # Risk weights from proposal
        self.weights = {
            'environmental': 0.40,
            'compliance': 0.30,
            'operational': 0.20,
            'reputational': 0.10
        }
    
    def calculate_risk(self, supplier_data):
        """
        Calculate risk score with breakdown
        
        supplier_data: {
            'emissions': float,
            'certifications': [],
            'audit_history': [],
            'financial_stability': float,
            ...
        }
        """
        # Calculate individual risk components
        env_risk = self._calculate_environmental_risk(supplier_data)
        comp_risk = self._calculate_compliance_risk(supplier_data)
        op_risk = self._calculate_operational_risk(supplier_data)
        rep_risk = self._calculate_reputational_risk(supplier_data)
        
        # Weighted total
        total_score = (
            env_risk * self.weights['environmental'] +
            comp_risk * self.weights['compliance'] +
            op_risk * self.weights['operational'] +
            rep_risk * self.weights['reputational']
        )
        
        # Categorize risk
        if total_score >= 85:
            category = 'CRITICAL'
            recommendations = [
                'Immediate audit required',
                'Consider alternative suppliers',
                'Implement monthly monitoring'
            ]
        elif total_score >= 70:
            category = 'HIGH'
            recommendations = [
                'Quarterly ESG review',
                'Request emission reduction plan',
                'Increase data collection frequency'
            ]
        elif total_score >= 50:
            category = 'MEDIUM'
            recommendations = [
                'Standard monitoring',
                'Annual ESG assessment'
            ]
        else:
            category = 'LOW'
            recommendations = ['Continue current practices']
        
        return {
            'total_score': round(total_score, 1),
            'breakdown': {
                'environmental': round(env_risk, 1),
                'compliance': round(comp_risk, 1),
                'operational': round(op_risk, 1),
                'reputational': round(rep_risk, 1)
            },
            'category': category,
            'recommendations': recommendations
        }
    
    def _calculate_environmental_risk(self, data):
        """40% weight - carbon intensity, emissions, renewables"""
        emissions = data.get('emissions', 50000)
        renewable_pct = data.get('renewable_pct', 10)
        
        # Normalize emissions (example threshold: 100,000 kg = high risk)
        emission_score = min(emissions / 100000 * 100, 100)
        
        # Lower renewable adoption = higher risk
        renewable_score = 100 - renewable_pct
        
        return (emission_score * 0.7 + renewable_score * 0.3)
    
    def _calculate_compliance_risk(self, data):
        """30% weight - certifications, audit history, violations"""
        certifications = data.get('certifications', [])
        violations = data.get('violations', 0)
        
        cert_score = 100 - len(certifications) * 15  # Each cert reduces risk
        violation_score = violations * 20  # Each violation adds risk
        
        return min(max(cert_score + violation_score, 0), 100)
    
    def _calculate_operational_risk(self, data):
        """20% weight - reliability, financial stability, location"""
        on_time_delivery = data.get('on_time_delivery_pct', 85)
        financial_score = data.get('financial_stability', 70)
        
        delivery_risk = 100 - on_time_delivery
        financial_risk = 100 - financial_score
        
        return (delivery_risk * 0.6 + financial_risk * 0.4)
    
    def _calculate_reputational_risk(self, data):
        """10% weight - media sentiment, controversies"""
        controversies = data.get('controversies', 0)
        media_sentiment = data.get('media_sentiment', 0.5)  # -1 to 1
        
        controversy_score = controversies * 25
        sentiment_score = (1 - media_sentiment) * 50
        
        return min(controversy_score + sentiment_score, 100)