import os
import uuid
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import re

from models.emission_forecasting import EmissionForecaster
from models.supplier_risk import SupplierRiskScorer
from models.optimization import ScenarioOptimizer
from models.regulatory_nlp import RegulatoryNLP, SupplierESGAnalyzer
from utils.data_processor import DataProcessor

app = Flask(__name__)
CORS(app)

SCENARIO_FILE = "./data/scenarios.json";

# Custom JSON encoder to handle NaN values
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN/Inf to null
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Helper function to clean data
def clean_json_data(data):
    """Recursively clean data to remove NaN values"""
    if isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    return data

# Initialize models
forecaster = EmissionForecaster()
risk_scorer = SupplierRiskScorer()
optimizer = ScenarioOptimizer()
data_processor = DataProcessor()
regulatory_nlp = RegulatoryNLP()
supplier_esg_analyzer = SupplierESGAnalyzer()

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'SSCA Backend API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'forecast': '/api/v1/forecast/emissions',
            'supplier_risk': '/api/v1/score/supplier',
            'optimization': '/api/v1/optimize/supply-chain',
            'regulation': '/api/v1/analyze/regulation',
            'suppliers': '/api/v1/suppliers'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'forecaster': forecaster.is_loaded,
            'risk_scorer': risk_scorer.is_loaded,
            'optimizer': True
        }
    })

@app.route('/api/v1/forecast/emissions', methods=['POST'])
def forecast_emissions():
    try:
        data = request.json
        facility_id = data.get('facility_id', 'DEMO')
        horizon_days = data.get('horizon_days', 30)
        
        # Generate forecast
        forecast_result = forecaster.predict(
            facility_id=facility_id,
            horizon=horizon_days
        )
        
        # Clean the data to remove NaN values
        cleaned_forecast = clean_json_data({
            'success': True,
            'forecast': forecast_result['predictions'],
            'confidence_intervals': forecast_result['confidence'],
            'dates': forecast_result['dates'],
            'metadata': {
                'facility_id': facility_id,
                'horizon_days': horizon_days,
                'generated_at': datetime.now().isoformat()
            }
        })
        
        return jsonify(cleaned_forecast)
    
    except Exception as e:
        print(f"Error in forecast_emissions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/score/supplier', methods=['POST'])
def score_supplier():
    try:
        data = request.json
        supplier_id = data.get('supplier_id')
        supplier_data = data.get('supplier_data', {})
        
        # Calculate risk score
        risk_result = risk_scorer.calculate_risk(supplier_data)
        
        # Clean the data
        cleaned_result = clean_json_data({
            'success': True,
            'supplier_id': supplier_id,
            'risk_score': risk_result['total_score'],
            'breakdown': risk_result['breakdown'],
            'risk_category': risk_result['category'],
            'recommendations': risk_result['recommendations']
        })
        
        return jsonify(cleaned_result)
    
    except Exception as e:
        print(f"Error in score_supplier: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/optimize/supply-chain', methods=['POST'])
def optimize_supply_chain():
    try:
        data = request.json
        scenario = data.get('scenario', {})
        
        # Extract scenario parameters
        modal_shift_pct = scenario.get('modal_shift_pct', 0)
        renewable_increase_pct = scenario.get('renewable_increase_pct', 0)
        
        # Baseline values
        baseline_emissions = 100000  # kg CO2e
        baseline_cost = 500000  # $
        
        # Calculate optimized values based on scenario parameters
        # Modal shift reduces emissions but increases cost
        modal_shift_impact = modal_shift_pct / 100
        emission_reduction_from_modal = modal_shift_impact * 0.15  # 15% max reduction
        cost_increase_from_modal = modal_shift_impact * 0.05  # 5% max cost increase
        
        # Renewable energy reduces emissions but increases cost more
        renewable_impact = renewable_increase_pct / 100
        emission_reduction_from_renewable = renewable_impact * 0.25  # 25% max reduction
        cost_increase_from_renewable = renewable_impact * 0.10  # 10% max cost increase
        
        # Total impact
        total_emission_reduction = emission_reduction_from_modal + emission_reduction_from_renewable
        total_cost_increase = cost_increase_from_modal + cost_increase_from_renewable
        
        # Calculate optimized values
        optimized_emissions = baseline_emissions * (1 - total_emission_reduction)
        optimized_cost = baseline_cost * (1 + total_cost_increase)
        
        emission_reduction_pct = total_emission_reduction * 100
        cost_change_pct = total_cost_increase * 100
        
        # Generate dynamic Pareto front
        pareto_solutions = []
        
        # Baseline point
        pareto_solutions.append({
            'name': 'Baseline',
            'cost': baseline_cost,
            'emissions': baseline_emissions,
            'modal_shift': 0,
            'renewable': 0
        })
        
        # Conservative approach (25% of targets)
        conservative_modal = modal_shift_pct * 0.25
        conservative_renewable = renewable_increase_pct * 0.25
        conservative_emission_reduction = (conservative_modal / 100 * 0.15 + conservative_renewable / 100 * 0.25)
        conservative_cost_increase = (conservative_modal / 100 * 0.05 + conservative_renewable / 100 * 0.10)
        
        pareto_solutions.append({
            'name': 'Conservative',
            'cost': int(baseline_cost * (1 + conservative_cost_increase)),
            'emissions': int(baseline_emissions * (1 - conservative_emission_reduction)),
            'modal_shift': int(conservative_modal),
            'renewable': int(conservative_renewable)
        })
        
        # Current optimized (actual scenario values)
        pareto_solutions.append({
            'name': f'Current ({modal_shift_pct}%/{renewable_increase_pct}%)',
            'cost': int(optimized_cost),
            'emissions': int(optimized_emissions),
            'modal_shift': modal_shift_pct,
            'renewable': renewable_increase_pct
        })
        
        # Balanced approach (150% of targets if not at max)
        if modal_shift_pct < 100 and renewable_increase_pct < 100:
            balanced_modal = min(modal_shift_pct * 1.5, 100)
            balanced_renewable = min(renewable_increase_pct * 1.5, 100)
            balanced_emission_reduction = (balanced_modal / 100 * 0.15 + balanced_renewable / 100 * 0.25)
            balanced_cost_increase = (balanced_modal / 100 * 0.05 + balanced_renewable / 100 * 0.10)
            
            pareto_solutions.append({
                'name': 'Balanced',
                'cost': int(baseline_cost * (1 + balanced_cost_increase)),
                'emissions': int(baseline_emissions * (1 - balanced_emission_reduction)),
                'modal_shift': int(balanced_modal),
                'renewable': int(balanced_renewable)
            })
        
        # Aggressive approach (maximum emission reduction)
        aggressive_emission_reduction = 0.40  # 40% max reduction
        aggressive_cost_increase = 0.18  # 18% cost increase
        
        pareto_solutions.append({
            'name': 'Aggressive',
            'cost': int(baseline_cost * (1 + aggressive_cost_increase)),
            'emissions': int(baseline_emissions * (1 - aggressive_emission_reduction)),
            'modal_shift': 100,
            'renewable': 100
        })
        
        supplier_changes = []
        if modal_shift_pct > 0:
            supplier_changes.append({
                'name': 'Alpha Steel',
                'supplier_id': 'S-001',
                'change_pct': int(modal_shift_pct * 0.8),
                'emission_impact': f'-{int(modal_shift_pct * 150)} tons CO2e',
                'note': 'Shifted from air to sea freight'
            })
        
        if renewable_increase_pct > 0:
            supplier_changes.append({
                'name': 'Beta Logistics',
                'supplier_id': 'S-002',
                'change_pct': int(renewable_increase_pct * 0.6),
                'emission_impact': f'-{int(renewable_increase_pct * 100)} tons CO2e',
                'note': 'Increased renewable energy usage'
            })
        
        cleaned_result = clean_json_data({
            'success': True,
            'scenario': scenario,
            'results': {
                'baseline_emissions': baseline_emissions,
                'optimized_emissions': int(optimized_emissions),
                'emission_reduction_pct': round(emission_reduction_pct, 2),
                'cost_change_pct': round(cost_change_pct, 2),
                'baseline_cost': baseline_cost,
                'optimized_cost': int(optimized_cost),
                'pareto_solutions': pareto_solutions,
                'supplier_changes': supplier_changes,
                'metadata': {
                    'modal_shift_pct': modal_shift_pct,
                    'renewable_increase_pct': renewable_increase_pct,
                    'calculated_at': datetime.now().isoformat()
                }
            }
        })
        
        return jsonify(cleaned_result)
    
    except Exception as e:
        print(f"Error in optimize_supply_chain: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/analyze/regulation', methods=['POST'])
def analyze_regulation():
    try:
        data = request.json
        regulation_text = data.get('regulation_text', '')
        analysis_type = data.get('analysis_type', 'regulation')  # 'regulation' or 'supplier_esg'
        
        if analysis_type == 'supplier_esg':
            # Analyze supplier ESG document
            supplier_id = data.get('supplier_id')
            analysis_result = supplier_esg_analyzer.analyze_supplier_esg(
                regulation_text, 
                supplier_id
            )
        else:
            # Analyze regulatory document
            analysis_result = regulatory_nlp.extract_regulatory_entities(regulation_text)
        
        return jsonify(analysis_result)
    
    except Exception as e:
        print(f"Error in analyze_regulation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

    try:
        data = request.json
        regulation_text = data.get('regulation_text', '')
        
        # Simplified NLP analysis
        entities = extract_regulatory_entities(regulation_text)
        
        return jsonify({
            'success': True,
            'entities': entities,
            'compliance_requirements': ['Scope 1/2/3 disclosure', 'Annual reporting'],
            'applicable_jurisdictions': ['EU', 'US']
        })
    
    except Exception as e:
        print(f"Error in analyze_regulation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/batch/analyze/regulations', methods=['POST'])
def batch_analyze_regulations():
    try:
        data = request.json
        documents = data.get('documents', [])
        analysis_type = data.get('analysis_type', 'regulation')
        
        results = []
        for doc in documents:
            if analysis_type == 'supplier_esg':
                result = supplier_esg_analyzer.analyze_supplier_esg(
                    doc.get('text', ''),
                    doc.get('supplier_id')
                )
            else:
                result = regulatory_nlp.extract_regulatory_entities(doc.get('text', ''))
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in batch_analyze_regulations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/chat/regulation', methods=['POST'])
def regulatory_chatbot():
    try:
        data = request.json
        question = data.get('question', '')
        
        # Sample knowledge base (in production, this would be vector database with embeddings)
        knowledge_base = {
            "csrd thresholds": {
                "answer": "Large undertakings with >750 employees OR >€100M net turnover must report Scope 1, 2, and material Scope 3 emissions starting fiscal year 2024.",
                "source": "CSRD ESRS E1, Section 4.2",
                "confidence": 0.95
            },
            "scope 3 reporting": {
                "answer": "Scope 3 emissions include 15 categories of indirect emissions from the value chain, including purchased goods, transportation, and waste.",
                "source": "GHG Protocol Corporate Standard",
                "confidence": 0.92
            },
            "renewable energy targets": {
                "answer": "The EU Renewable Energy Directive requires 42.5% renewable energy by 2030, with indicative target of 45%.",
                "source": "RED III Directive 2023/2413",
                "confidence": 0.88
            }
        }
        
        # Simple keyword matching (in production, use embeddings + similarity search)
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        for key, value in knowledge_base.items():
            score = sum(1 for word in key.split() if word in question_lower)
            if score > best_score:
                best_score = score
                best_match = value
        
        if best_match:
            response = {
                'success': True,
                'question': question,
                'answer': best_match['answer'],
                'source': best_match.get('source', ''),
                'confidence': best_match.get('confidence', 0),
                'suggested_queries': [
                    "What are the CSRD reporting thresholds?",
                    "How to calculate Scope 3 emissions?",
                    "What are renewable energy targets for 2030?"
                ]
            }
        else:
            response = {
                'success': True,
                'question': question,
                'answer': "I couldn't find specific information about that query. Please try asking about CSRD thresholds, Scope 3 reporting, or renewable energy targets.",
                'source': '',
                'confidence': 0.3,
                'suggested_queries': []
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in regulatory_chatbot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def extract_regulatory_entities(text):
    """Simplified entity extraction"""
    entities = {
        'thresholds': [],
        'deadlines': [],
        'scopes': [],
        'penalties': []
    }
    
    if 'Scope 1' in text or 'Scope 2' in text or 'Scope 3' in text:
        entities['scopes'].append('Scope 1, 2, and 3')
    
    if '750 employees' in text:
        entities['thresholds'].append('750 employees')
    
    return entities

@app.route('/api/v1/chat/reporting', methods=['POST'])
def reporting_chatbot():
    try:
        data = request.json
        question = data.get('question', '')
        
        # Sample knowledge base for reporting
        reporting_knowledge_base = {
            "generate pdf report": {
                "answer": "I'll help you generate a PDF report. Please specify the reporting period and framework. For example: 'Generate a PDF report for 2024 using CSRD framework'.",
                "action": "generate_pdf",
                "parameters": ["period", "framework"],
                "confidence": 0.95
            },
            "export to excel": {
                "answer": "I can export data to Excel format. Please specify what data you need: supplier emissions, scope breakdown, or compliance data. Example: 'Export supplier emissions data to Excel'.",
                "action": "export_excel",
                "parameters": ["data_type"],
                "confidence": 0.92
            },
            "generate docx": {
                "answer": "I'll create a DOCX report for you. Please specify the reporting period and framework. Example: 'Generate a DOCX report for Q4 2024 using GRI standards'.",
                "action": "generate_docx",
                "parameters": ["period", "framework"],
                "confidence": 0.90
            },
            "compliance report": {
                "answer": "I can generate a compliance report showing regulatory adherence. Specify the jurisdiction (EU, US, etc.) and reporting period.",
                "action": "compliance_report",
                "parameters": ["jurisdiction", "period"],
                "confidence": 0.88
            },
            "supplier emissions report": {
                "answer": "I'll generate a supplier emissions report with Scope 1, 2, and 3 breakdowns. Please specify the time period and format (PDF, Excel, or DOCX).",
                "action": "supplier_emissions_report",
                "parameters": ["period", "format"],
                "confidence": 0.85
            }
        }
        
        # Simple keyword matching (in production, use NLP)
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        for key, value in reporting_knowledge_base.items():
            score = sum(1 for word in key.split() if word in question_lower)
            if score > best_score:
                best_score = score
                best_match = value
        
        if best_match:
            response = {
                'success': True,
                'question': question,
                'answer': best_match['answer'],
                'action': best_match.get('action', ''),
                'parameters': best_match.get('parameters', []),
                'confidence': best_match.get('confidence', 0),
                'suggested_queries': [
                    "Generate a PDF report for 2024 using CSRD",
                    "Export supplier emissions data to Excel",
                    "Generate a DOCX report for Q4 2023",
                    "Create a compliance report for EU regulations"
                ]
            }
        else:
            response = {
                'success': True,
                'question': question,
                'answer': "I'm here to help with report generation and data export. You can ask me to generate PDF, Excel, or DOCX reports, or ask about specific data exports.",
                'action': '',
                'parameters': [],
                'confidence': 0.3,
                'suggested_queries': []
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in reporting_chatbot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/generate/report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        report_type = data.get('report_type', '')
        parameters = data.get('parameters', {})
        
        # Simulate report generation
        report_id = f"REP-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        
        # Different report types
        report_templates = {
            'pdf': {
                'file_type': 'pdf',
                'size_mb': 2.5,
                'pages': 24,
                'sections': ['Executive Summary', 'Emissions Data', 'Supplier Analysis', 'Compliance Status']
            },
            'excel': {
                'file_type': 'xlsx',
                'size_mb': 1.8,
                'sheets': ['Summary', 'Raw Data', 'Calculations', 'Visualizations'],
                'rows': 1500
            },
            'docx': {
                'file_type': 'docx',
                'size_mb': 1.2,
                'pages': 18,
                'sections': ['Cover Page', 'Table of Contents', 'Main Report', 'Appendices']
            }
        }
        
        report_info = report_templates.get(report_type, report_templates['pdf'])
        
        return jsonify({
            'success': True,
            'report_id': report_id,
            'report_type': report_type,
            'parameters': parameters,
            'download_url': f'/api/v1/download/{report_id}',
            'generated_at': datetime.now().isoformat(),
            'estimated_time_seconds': 15,
            'file_info': report_info,
            'message': f'Report generation started. Your report ID is {report_id}. It will be ready in approximately 15 seconds.'
        })
    
    except Exception as e:
        print(f"Error in generate_report: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/report/status/<report_id>', methods=['GET'])
def get_report_status(report_id):
    try:
        # In production, check database or file system
        return jsonify({
            'success': True,
            'report_id': report_id,
            'status': 'completed',
            'progress': 100,
            'download_url': f'/api/v1/download/{report_id}',
            'estimated_completion': datetime.now().isoformat(),
            'file_size': '2.5 MB',
            'formats_available': ['PDF', 'DOCX', 'Excel']
        })
    except Exception as e:
        print(f"Error in get_report_status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/download/<report_id>', methods=['GET'])
def download_report(report_id):
    try:
        report_type = request.args.get('type', 'pdf')
        
        # Determine file extension and MIME type
        mime_types = {
            'pdf': ('application/pdf', 'pdf'),
            'xlsx': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'),
            'excel': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'),
            'docx': ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'docx'),
            'csv': ('text/csv', 'csv'),
            'txt': ('text/plain', 'txt')
        }
        
        mime_type, extension = mime_types.get(report_type, ('application/octet-stream', 'bin'))
        
        # Create sample content based on type
        if report_type == 'pdf':
            # Create a simple PDF
            from reportlab.pdfgen import canvas
            import io
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 750, f"Sustainability Report: {report_id}")
            c.setFont("Helvetica", 12)
            c.drawString(100, 720, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(100, 700, "Powered by SSCA AI Assistant")
            c.drawString(100, 650, "Summary:")
            c.drawString(120, 630, "- Total Emissions: 125,450 tons CO2e")
            c.drawString(120, 610, "- Scope 1: 45,200 tons CO2e")
            c.drawString(120, 590, "- Scope 2: 35,800 tons CO2e")
            c.drawString(120, 570, "- Scope 3: 44,450 tons CO2e")
            c.drawString(120, 550, "- Renewable Energy: 28%")
            c.save()
            buffer.seek(0)
            data = buffer.getvalue()
            
        elif report_type in ['xlsx', 'excel']:
            # Create CSV data (simplified Excel)
            import csv
            import io
            
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(['Supplier', 'Country', 'Scope1_Emissions', 'Scope2_Emissions', 'Scope3_Emissions', 'Risk_Score', 'Renewable_Pct'])
            writer.writerow(['Alpha Steel', 'CN', 45200, 35800, 44450, 88, 15])
            writer.writerow(['Beta Logistics', 'ID', 12000, 15000, 25400, 72, 25])
            writer.writerow(['Gamma Parts', 'VN', 8000, 12000, 15600, 45, 40])
            writer.writerow(['Delta Energy', 'US', 75000, 85000, 106000, 95, 8])
            data = buffer.getvalue().encode('utf-8')
            
        else:
            # Create text content for other types
            content = f"""Sustainability Report
            Report ID: {report_id}
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Type: {report_type}

            SUMMARY:
            • Total Emissions: 125,450 tons CO2e
            • Scope 1: 45,200 tons CO2e
            • Scope 2: 35,800 tons CO2e
            • Scope 3: 44,450 tons CO2e
            • Renewable Energy: 28%
            • High-risk Suppliers: 2

            This report was generated by SSCA AI Assistant.
            For more information, visit the reporting dashboard.
            """
            data = content.encode('utf-8')
        
        # Create response with proper headers
        response = Response(
            data,
            mimetype=mime_type,
            headers={
                'Content-Disposition': f'attachment; filename="{report_id}.{extension}"',
                'Content-Length': len(data),
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
        return response
        
    except Exception as e:
        print(f"Error in download_report: {str(e)}")
        # Return error as JSON
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/reports/generated', methods=['GET'])
def list_generated_reports():
    try:
        # In production, fetch from database
        reports = [
            {
                'id': 'REP-20240115-001',
                'name': 'CSRD Compliance Report 2024',
                'type': 'pdf',
                'size': '2.5 MB',
                'created_at': '2024-01-15T10:30:00',
                'status': 'completed',
                'download_url': '/api/v1/download/REP-20240115-001'
            },
            {
                'id': 'REP-20240110-002',
                'name': 'Supplier Emissions Dashboard',
                'type': 'xlsx',
                'size': '1.8 MB',
                'created_at': '2024-01-10T14:45:00',
                'status': 'completed',
                'download_url': '/api/v1/download/REP-20240110-002'
            },
            {
                'id': 'REP-20240105-003',
                'name': 'Sustainability Performance Q4 2023',
                'type': 'docx',
                'size': '1.2 MB',
                'created_at': '2024-01-05T09:15:00',
                'status': 'completed',
                'download_url': '/api/v1/download/REP-20240105-003'
            }
        ]
        
        return jsonify({
            'success': True,
            'reports': reports,
            'count': len(reports),
            'total_size': '5.5 MB'
        })
    
    except Exception as e:
        print(f"Error in list_generated_reports: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

suppliers_db = [
    {
        'id': 'S-001',
        'name': 'Alpha Steel',
        'country': 'CN',
        'emissions': 125450,
        'risk': 88,
        'renewable_pct': 15,
        'tier': 'Tier-1',
        'category': 'Manufacturing',
        'status': 'Active',
        'created_at': '2024-01-15',
        'last_updated': '2024-01-15'
    },
    {
        'id': 'S-002',
        'name': 'Beta Logistics',
        'country': 'ID',
        'emissions': 52400,
        'risk': 72,
        'renewable_pct': 25,
        'tier': 'Tier-1',
        'category': 'Logistics',
        'status': 'Active',
        'created_at': '2024-01-10',
        'last_updated': '2024-01-10'
    },
    {
        'id': 'S-003',
        'name': 'Gamma Parts',
        'country': 'VN',
        'emissions': 35600,
        'risk': 45,
        'renewable_pct': 40,
        'tier': 'Tier-2',
        'category': 'Manufacturing',
        'status': 'Active',
        'created_at': '2024-01-05',
        'last_updated': '2024-01-05'
    },
    {
        'id': 'S-004',
        'name': 'Delta Energy',
        'country': 'US',
        'emissions': 266000,
        'risk': 95,
        'renewable_pct': 8,
        'tier': 'Tier-1',
        'category': 'Energy',
        'status': 'Active',
        'created_at': '2024-01-12',
        'last_updated': '2024-01-12'
    }
]

# Helper function for safe integer conversion
def safe_int(value, default=0):
    """Safely convert value to integer"""
    if value is None or value == '':
        return default
    try:
        # Handle string values
        if isinstance(value, str):
            # Remove any non-numeric characters except minus sign and decimal
            import re
            cleaned = re.sub(r'[^\d.-]', '', value)
            if cleaned and cleaned.replace('.', '', 1).replace('-', '', 1).isdigit():
                return int(float(cleaned))
            return default
        # Handle numeric types
        return int(value)
    except (ValueError, TypeError):
        return default

# Helper function for safe float conversion
def safe_float(value, default=0):
    """Safely convert value to float"""
    if value is None or value == '':
        return default
    try:
        # Handle string values
        if isinstance(value, str):
            # Remove any non-numeric characters except minus sign and decimal
            import re
            cleaned = re.sub(r'[^\d.-]', '', value)
            if cleaned:
                return float(cleaned)
            return default
        # Handle numeric types
        return float(value)
    except (ValueError, TypeError):
        return default

@app.route('/api/v1/suppliers', methods=['GET'])
def get_suppliers():
    try:
        return jsonify({
            'success': True,
            'suppliers': suppliers_db,
            'count': len(suppliers_db),
            'total_emissions': sum(s.get('emissions', 0) for s in suppliers_db),
            'avg_risk': sum(s.get('risk', 0) for s in suppliers_db) / len(suppliers_db) if suppliers_db else 0
        })
    except Exception as e:
        print(f"Error in get_suppliers: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/suppliers', methods=['POST'])
def add_supplier():
    try:
        data = request.json
        print(f"[ADD SUPPLIER] Received data: {data}")
        
        # Generate unique ID
        supplier_id = f"S-{len(suppliers_db) + 1:03d}"
        
        new_supplier = {
            'id': supplier_id,
            'name': data.get('name', 'New Supplier'),
            'country': data.get('country', 'Unknown'),
            'emissions': safe_int(data.get('emissions'), 0),
            'risk': safe_int(data.get('risk'), 50),
            'renewable_pct': safe_int(data.get('renewable_pct'), 0),
            'tier': data.get('tier', 'Tier-1'),
            'category': data.get('category', 'Other'),
            'status': data.get('status', 'Active'),
            'created_at': datetime.now().strftime('%Y-%m-%d'),
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
        
        suppliers_db.append(new_supplier)
        
        print(f"[ADD SUPPLIER] Added: {new_supplier}")
        print(f"[ADD SUPPLIER] Total suppliers now: {len(suppliers_db)}")
        
        return jsonify({
            'success': True,
            'supplier': new_supplier,
            'message': f'Supplier {supplier_id} added successfully'
        })
        
    except Exception as e:
        print(f"[ERROR] in add_supplier: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/suppliers/<supplier_id>', methods=['PUT'])
def update_supplier(supplier_id):
    try:
        data = request.json
        print(f"[UPDATE SUPPLIER] Updating {supplier_id} with data: {data}")
        
        for i, supplier in enumerate(suppliers_db):
            if supplier['id'] == supplier_id:
                
                updates = {}
                if 'name' in data:
                    updates['name'] = data['name']
                if 'country' in data:
                    updates['country'] = data['country']
                if 'emissions' in data:
                    updates['emissions'] = safe_int(data['emissions'], supplier['emissions'])
                if 'risk' in data:
                    updates['risk'] = safe_int(data['risk'], supplier['risk'])
                if 'renewable_pct' in data:
                    updates['renewable_pct'] = safe_int(data['renewable_pct'], supplier['renewable_pct'])
                if 'tier' in data:
                    updates['tier'] = data['tier']
                if 'category' in data:
                    updates['category'] = data['category']
                if 'status' in data:
                    updates['status'] = data['status']
                
                suppliers_db[i].update(updates)
                suppliers_db[i]['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                
                print(f"[UPDATE SUPPLIER] Updated: {suppliers_db[i]}")
                
                return jsonify({
                    'success': True,
                    'supplier': suppliers_db[i],
                    'message': f'Supplier {supplier_id} updated successfully'
                })
        
        print(f"[UPDATE SUPPLIER] Supplier {supplier_id} not found")
        return jsonify({'success': False, 'error': 'Supplier not found'}), 404
        
    except Exception as e:
        print(f"[ERROR] in update_supplier: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/suppliers/<supplier_id>', methods=['DELETE'])
def delete_supplier(supplier_id):
    try:
        global suppliers_db
        
        for i, supplier in enumerate(suppliers_db):
            if supplier['id'] == supplier_id:
                deleted_supplier = suppliers_db.pop(i)
                return jsonify({
                    'success': True,
                    'supplier': deleted_supplier,
                    'message': f'Supplier {supplier_id} deleted successfully'
                })
        
        return jsonify({'success': False, 'error': 'Supplier not found'}), 404
        
    except Exception as e:
        print(f"Error in delete_supplier: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/suppliers/upload', methods=['POST'])
def upload_supplier_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        file_type = request.form.get('file_type', 'csv')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read the file
        content = file.read().decode('utf-8')
        
        # Parse based on file type
        if file_type == 'csv':
            import csv
            import io
            
            csv_reader = csv.DictReader(io.StringIO(content))
            uploaded_suppliers = []
            
            for row in csv_reader:
                supplier_id = f"S-{len(suppliers_db) + len(uploaded_suppliers) + 1:03d}"
                new_supplier = {
                    'id': supplier_id,
                    'name': row.get('Name', row.get('name', 'Unknown')),
                    'country': row.get('Country', row.get('country', 'Unknown')),
                    'emissions': int(row.get('Emissions', row.get('emissions', 0)) or 0),
                    'risk': int(row.get('Risk', row.get('risk', 50)) or 50),
                    'renewable_pct': int(row.get('Renewable', row.get('renewable_pct', 0)) or 0),
                    'tier': row.get('Tier', row.get('tier', 'Tier-1')),
                    'category': row.get('Category', row.get('category', 'Other')),
                    'status': 'Active',
                    'created_at': datetime.now().strftime('%Y-%m-%d'),
                    'last_updated': datetime.now().strftime('%Y-%m-%d')
                }
                uploaded_suppliers.append(new_supplier)
            
            # Add to database
            suppliers_db.extend(uploaded_suppliers)
            
            return jsonify({
                'success': True,
                'uploaded_count': len(uploaded_suppliers),
                'suppliers': uploaded_suppliers,
                'message': f'Successfully uploaded {len(uploaded_suppliers)} suppliers'
            })
        
        else:
            return jsonify({'success': False, 'error': f'Unsupported file type: {file_type}'}), 400
        
    except Exception as e:
        print(f"Error in upload_supplier_file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/suppliers/analytics', methods=['GET'])
def get_supplier_analytics():
    try:
        if not suppliers_db:
            return jsonify({'success': False, 'error': 'No suppliers data'}), 404
        
        # Helper function to safely convert to integer
        def safe_int(value, default=0):
            try:
                if isinstance(value, str):
                    # Try to extract numbers from strings
                    import re
                    numbers = re.findall(r'\d+', value)
                    if numbers:
                        return int(numbers[0])
                    return default
                return int(value)
            except (ValueError, TypeError):
                return default
        
        # Helper function to safely convert to float
        def safe_float(value, default=0):
            try:
                if isinstance(value, str):
                    # Try to extract numbers from strings
                    import re
                    numbers = re.findall(r'\d+\.?\d*', value)
                    if numbers:
                        return float(numbers[0])
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Calculate analytics with safe conversions
        total_emissions = sum(safe_int(s.get('emissions', 0)) for s in suppliers_db)
        avg_risk = sum(safe_float(s.get('risk', 0)) for s in suppliers_db) / len(suppliers_db) if suppliers_db else 0
        avg_renewable = sum(safe_float(s.get('renewable_pct', 0)) for s in suppliers_db) / len(suppliers_db) if suppliers_db else 0
        
        # Count by tier
        tier_counts = {}
        for s in suppliers_db:
            tier = s.get('tier', 'Unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Count by country
        country_counts = {}
        for s in suppliers_db:
            country = s.get('country', 'Unknown')
            country_counts[country] = country_counts.get(country, 0) + 1
        
        # Risk distribution
        risk_distribution = {
            'critical': sum(1 for s in suppliers_db if safe_float(s.get('risk', 0)) >= 85),
            'high': sum(1 for s in suppliers_db if 70 <= safe_float(s.get('risk', 0)) < 85),
            'medium': sum(1 for s in suppliers_db if 50 <= safe_float(s.get('risk', 0)) < 70),
            'low': sum(1 for s in suppliers_db if safe_float(s.get('risk', 0)) < 50)
        }
        
        # Get top emitters
        sorted_emitters = sorted(
            suppliers_db, 
            key=lambda x: safe_int(x.get('emissions', 0)), 
            reverse=True
        )[:5]
        
        # Get high risk suppliers
        high_risk_suppliers = [
            {**s, 'emissions': safe_int(s.get('emissions', 0))}
            for s in suppliers_db 
            if safe_float(s.get('risk', 0)) >= 70
        ]
        
        return jsonify({
            'success': True,
            'analytics': {
                'total_suppliers': len(suppliers_db),
                'total_emissions': total_emissions,
                'avg_risk': round(avg_risk, 2),
                'avg_renewable': round(avg_renewable, 2),
                'tier_distribution': tier_counts,
                'country_distribution': country_counts,
                'risk_distribution': risk_distribution,
                'top_emitters': sorted_emitters,
                'high_risk_suppliers': high_risk_suppliers
            }
        })
        
    except Exception as e:
        print(f"[ERROR] in get_supplier_analytics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Ensure the file exists
def ensure_scenario_file():
    try:
        with open(SCENARIO_FILE, "r") as f:
            pass   # file exists, do nothing
    except FileNotFoundError:
        with open(SCENARIO_FILE, "w") as f:
            json.dump([], f)


def load_scenario_file():
    ensure_scenario_file()
    with open(SCENARIO_FILE, "r") as f:
        return json.load(f)


def save_scenario_file(data):
    with open(SCENARIO_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/api/v1/scenarios", methods=["GET"])
def get_scenarios():
    try:
        data = load_scenario_file()
        return jsonify({"success": True, "scenarios": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/v1/scenarios", methods=["POST"])
def save_scenario():
    try:
        new_scenario = request.json

        # Auto-generate an ID if missing
        if "id" not in new_scenario or not new_scenario["id"]:
            new_scenario["id"] = str(uuid.uuid4())

        data = load_scenario_file()

        # Remove existing scenario with same ID
        data = [s for s in data if s.get("id") != new_scenario["id"]]

        data.append(new_scenario)
        save_scenario_file(data)

        return jsonify({"success": True, "id": new_scenario["id"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/v1/scenarios/<scenario_id>", methods=["DELETE"])
def delete_scenario(scenario_id):
    try:
        data = load_scenario_file()
        data = [s for s in data if s.get("id") != scenario_id]
        save_scenario_file(data)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': {
            'root': '/',
            'health': '/health',
            'forecast': '/api/v1/forecast/emissions',
            'supplier_risk': '/api/v1/score/supplier',
            'optimization': '/api/v1/optimize/supply-chain',
            'regulation': '/api/v1/analyze/regulation',
            'suppliers': '/api/v1/suppliers'
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    print("Starting SSCA Backend Server...")
    print("Server running on http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)