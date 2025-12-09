import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import TruncatedNormal, Normal

class RegulatoryNLPModel(nn.Cell):
    """Transformer-based NER model for regulatory document analysis"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_attention_heads=12, 
                 num_hidden_layers=6, num_labels=7, max_position_embeddings=512):
        super(RegulatoryNLPModel, self).__init__()
        
        # Entity labels (aligned with proposal)
        self.entity_labels = [
            "THRESHOLD",      # e.g., "750 employees", "€100M turnover"
            "DEADLINE",       # e.g., "fiscal year 2024", "by December 31, 2025"
            "PENALTY",        # e.g., "fine up to €500,000"
            "SCOPE",          # e.g., "Scope 1, 2, and 3"
            "ENTITY",         # e.g., "Large undertakings", "manufacturing facilities"
            "EMISSION_VALUE", # e.g., "45,200 tons CO2e" (from supplier reports)
            "RENEWABLE_PCT"   # e.g., "28% renewable energy"
        ]
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer layers
        self.encoder_layers = nn.CellList([
            TransformerEncoderLayer(hidden_size, num_attention_heads)
            for _ in range(num_hidden_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm((hidden_size,), epsilon=1e-12)
        
        # Classifier head for NER
        self.classifier = nn.Dense(hidden_size, num_labels)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def construct(self, input_ids, attention_mask=None):
        """Forward pass through the model"""
        seq_length = input_ids.shape[1]
        position_ids = ops.arange(seq_length).broadcast_to((input_ids.shape[0], seq_length))
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Apply classifier
        logits = self.classifier(hidden_states)
        
        return logits

class TransformerEncoderLayer(nn.Cell):
    """Single transformer encoder layer"""
    
    def __init__(self, hidden_size, num_attention_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_attention_heads)
        self.attention_layer_norm = nn.LayerNorm((hidden_size,), epsilon=1e-12)
        self.feed_forward = FeedForward(hidden_size)
        self.output_layer_norm = nn.LayerNorm((hidden_size,), epsilon=1e-12)
        self.dropout = nn.Dropout(0.1)
        
    def construct(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.self_attention(hidden_states, hidden_states, hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = self.output_layer_norm(hidden_states + ff_output)
        
        return hidden_states

class MultiHeadAttention(nn.Cell):
    """Multi-head attention mechanism"""
    
    def __init__(self, hidden_size, num_attention_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        self.query = nn.Dense(hidden_size, hidden_size)
        self.key = nn.Dense(hidden_size, hidden_size)
        self.value = nn.Dense(hidden_size, hidden_size)
        self.output = nn.Dense(hidden_size, hidden_size)
        
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(0.1)
        
    def construct(self, query, key, value, attention_mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        key = key.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        value = value.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        
        # Transpose for attention computation
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = ops.matmul(query, key.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / ops.sqrt(Tensor(self.attention_head_size, ms.float32))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, -1)
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = ops.matmul(attention_probs, value)
        context = context.transpose(0, 2, 1, 3)
        context = context.view(batch_size, -1, self.num_attention_heads * self.attention_head_size)
        
        # Output projection
        output = self.output(context)
        
        return output

class FeedForward(nn.Cell):
    """Feed-forward neural network"""
    
    def __init__(self, hidden_size):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Dense(hidden_size, hidden_size * 4)
        self.dense2 = nn.Dense(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def construct(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states

class RegulatoryNLP:
    """Main regulatory NLP engine with multilingual support"""
    
    def __init__(self, model_path=None):
        # Initialize model (in production, load from trained checkpoint)
        self.model = RegulatoryNLPModel()
        
        # Language detection patterns
        self.language_patterns = {
            'en': r'\b(the|and|of|to|a|in|that|is|for|it)\b',
            'de': r'\b(der|die|das|und|in|den|von|zu|das|mit)\b',
            'fr': r'\b(le|la|les|et|de|un|une|des|en|que)\b',
            'es': r'\b(el|la|los|las|y|de|que|en|un|una)\b',
            'zh': r'[\u4e00-\u9fff]',
            'id': r'\b(dan|yang|di|untuk|dari|dengan|ini|itu|tidak|akan)\b'
        }
        
        # Entity extraction patterns (regex-based as fallback)
        self.entity_patterns = {
            'THRESHOLD': [
                r'\b(\d{1,3}(?:,\d{3})*)\s*(employees?|staff|workers?|turnover|revenue|sales)\b',
                r'[€$£]\s*(\d+(?:\.\d+)?)\s*(million|billion|M|B)\b',
                r'\b(more than|at least|greater than|over)\s+(\d{1,3}(?:,\d{3})*)\b'
            ],
            'DEADLINE': [
                r'\b(fiscal year|FY)\s*(\d{4})\b',
                r'\b(by|until|before)\s+(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})\b',
                r'\b(starting|effective|from)\s+(\w+\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})\b'
            ],
            'PENALTY': [
                r'\b(fine|penalty|sanction)\s+(of\s+)?[€$£]?\s*(\d{1,3}(?:,\d{3})*)\b',
                r'\b(up to|maximum)\s+[€$£]?\s*(\d{1,3}(?:,\d{3})*)\s*(fine|penalty)\b',
                r'\b(imprisonment|jail)\s+(of\s+)?(\d+)\s*(years?|months?)\b'
            ],
            'SCOPE': [
                r'\b(Scope\s*[123])\b',
                r'\b(scope\s+[123])\b',
                r'\b(direct|indirect|value\s+chain)\s+emissions?\b'
            ],
            'EMISSION_VALUE': [
                r'\b(\d{1,3}(?:,\d{3})*)\s*(tons?|tonnes?|kg|metric\s+tons?)\s*(CO2e?|CO2\s+equivalent)\b',
                r'\b(\d+(?:\.\d+)?)\s*(million|billion)?\s*tons?\s*CO2\b',
                r'\b(emissions?|CO2)\s+of\s+(\d{1,3}(?:,\d{3})*)\s*(tons?|kg)\b'
            ],
            'RENEWABLE_PCT': [
                r'\b(\d{1,3})%\s*(renewable|green|clean)\s*(energy|electricity|power)\b',
                r'\b(renewable|green)\s*(energy|power)\s*(\d{1,3})%\b',
                r'\b(\d{1,3})%\s*from\s*(renewable|solar|wind|hydro)\b'
            ]
        }
        
        # Jurisdiction detection
        self.jurisdiction_keywords = {
            'EU': ['European Union', 'EU', 'European Commission', 'Brussels', 'CSRD', 'ESRS'],
            'US': ['United States', 'US', 'USA', 'SEC', 'EPA', 'Federal', 'Washington'],
            'UK': ['United Kingdom', 'UK', 'Britain', 'London', 'BEIS'],
            'CN': ['China', 'Chinese', 'Beijing', 'CCP', 'CNCA'],
            'ID': ['Indonesia', 'Indonesian', 'Jakarta', 'IDX', 'OJK'],
            'IN': ['India', 'Indian', 'New Delhi', 'BSE', 'SEBI'],
            'JP': ['Japan', 'Japanese', 'Tokyo', 'METI', 'TSE']
        }
        
        # Initialize tokenizer (simplified version)
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        """Build a simple vocabulary for demonstration"""
        vocab = {}
        # Add common words
        words = [
            'emissions', 'carbon', 'scope', 'reporting', 'disclosure',
            'threshold', 'deadline', 'penalty', 'regulation', 'compliance',
            'sustainable', 'environmental', 'social', 'governance', 'esg',
            'renewable', 'energy', 'electricity', 'power', 'generation',
            'tons', 'kg', 'co2', 'equivalent', 'metric', 'reduction',
            'target', 'objective', 'goal', 'achievement', 'progress'
        ]
        
        for i, word in enumerate(words):
            vocab[word] = i + 1  # Start from 1 (0 reserved for padding)
            
        return vocab
    
    def detect_language(self, text):
        """Detect language of the text"""
        text_lower = text.lower()
        scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            scores[lang] = len(matches)
            
        # Return language with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'en'  # Default to English
    
    def identify_jurisdiction(self, text):
        """Identify jurisdiction based on keywords"""
        text_lower = text.lower()
        jurisdictions = []
        
        for jurisdiction, keywords in self.jurisdiction_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    jurisdictions.append(jurisdiction)
                    break
                    
        return list(set(jurisdictions)) if jurisdictions else ['Global']
    
    def tokenize(self, text, max_length=512):
        """Simple tokenization for demonstration"""
        # Split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Convert to token IDs
        token_ids = []
        for word in words[:max_length]:
            token_ids.append(self.vocab.get(word, 0))  # 0 for unknown
        
        # Pad or truncate to max_length
        if len(token_ids) < max_length:
            token_ids += [0] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
            
        return token_ids
    
    def extract_entities_regex(self, text):
        """Extract entities using regex patterns (fallback method)"""
        entities = {
            'thresholds': [],
            'deadlines': [],
            'penalties': [],
            'scopes': [],
            'entities': [],
            'emission_values': [],
            'renewable_pcts': []
        }
        
        text_lower = text.lower()
        
        # Extract using regex patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(0)
                    
                    if entity_type == 'THRESHOLD':
                        entities['thresholds'].append(value)
                    elif entity_type == 'DEADLINE':
                        entities['deadlines'].append(value)
                    elif entity_type == 'PENALTY':
                        entities['penalties'].append(value)
                    elif entity_type == 'SCOPE':
                        entities['scopes'].append(value)
                    elif entity_type == 'EMISSION_VALUE':
                        entities['emission_values'].append(value)
                    elif entity_type == 'RENEWABLE_PCT':
                        entities['renewable_pcts'].append(value)
        
        return entities
    
    def extract_regulatory_entities(self, text, use_model=True):
        """Main method to extract regulatory entities from text"""
        
        language = self.detect_language(text)
        
        jurisdictions = self.identify_jurisdiction(text)
        
        if use_model:
            entities = self._extract_with_model(text)
        else:
            entities = self.extract_entities_regex(text)
        
        compliance_reqs = self._extract_compliance_requirements(text)
        
        regulation_type = self._classify_regulation(text)
        
        dates = self._extract_dates(text)
        
        return {
            'success': True,
            'text': text[:500] + "..." if len(text) > 500 else text,
            'metadata': {
                'language': language,
                'jurisdictions': jurisdictions,
                'regulation_type': regulation_type,
                'word_count': len(text.split()),
                'extraction_date': datetime.now().isoformat()
            },
            'entities': entities,
            'compliance_requirements': compliance_reqs,
            'dates': dates,
            'summary': self._generate_summary(entities, compliance_reqs)
        }
    
    def _extract_with_model(self, text):
        """Extract entities using the trained model"""

        tokens = self.tokenize(text)
        
        input_tensor = Tensor([tokens], ms.int32)
        
        entities = self.extract_entities_regex(text)  # Fallback for now
        
        return entities
    
    def _extract_compliance_requirements(self, text):
        """Extract compliance requirements from text"""
        requirements = []
        
        # Look for requirement patterns
        requirement_patterns = [
            r'\b(must|shall|required to|obligated to|mandatory)\s+([^.!?]+[.!?])',
            r'\b(shall\s+disclose|must\s+report|required\s+to\s+publish)\s+([^.!?]+[.!?])',
            r'\b(compliance with|in accordance with|pursuant to)\s+([^.!?]+[.!?])'
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                requirements.append(match[1].strip())
        
        return list(set(requirements))[:10]  # Return top 10 unique requirements
    
    def _classify_regulation(self, text):
        """Classify the type of regulation"""
        text_lower = text.lower()
        
        regulation_types = {
            'CSRD': ['csrd', 'corporate sustainability reporting directive', 'esrs'],
            'SEC': ['sec', 'securities and exchange commission', 'climate rules'],
            'TCFD': ['tcfd', 'task force on climate-related financial disclosures'],
            'GRI': ['gri', 'global reporting initiative', 'gri 305'],
            'CDP': ['cdp', 'carbon disclosure project'],
            'ISO': ['iso 14064', 'iso 14001', 'international organization for standardization'],
            'GHG': ['ghg protocol', 'greenhouse gas protocol']
        }
        
        for reg_type, keywords in regulation_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return reg_type
        
        return 'UNKNOWN'
    
    def _extract_dates(self, text):
        """Extract dates from text"""
        date_patterns = [
            r'\b(\d{4})\b',  # Year
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY
            r'\b(\d{1,2}\s+\w+\s+\d{4})\b',  # 1 January 2024
            r'\b(fiscal year|FY)\s*(\d{4})\b',
            r'\b(effective|starting|from)\s+(\w+\s+\d{4}|\d{4})\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    dates.append(match[-1])  # Get the last element of tuple
                else:
                    dates.append(match)
        
        return list(set(dates))
    
    def _generate_summary(self, entities, compliance_reqs):
        """Generate a summary of the regulatory analysis"""
        summary_parts = []
        
        if entities['thresholds']:
            summary_parts.append(f"Found {len(entities['thresholds'])} regulatory thresholds")
        
        if entities['deadlines']:
            summary_parts.append(f"Identified {len(entities['deadlines'])} compliance deadlines")
        
        if entities['penalties']:
            summary_parts.append(f"Found {len(entities['penalties'])} penalty provisions")
        
        if entities['scopes']:
            scopes = ', '.join(set([s.replace('scope', '').strip() for s in entities['scopes']]))
            summary_parts.append(f"Applicable to Scope {scopes}")
        
        if compliance_reqs:
            summary_parts.append(f"{len(compliance_reqs)} specific compliance requirements")
        
        if summary_parts:
            return "; ".join(summary_parts)
        
        return "No specific regulatory entities identified"

# For supplier ESG document analysis
class SupplierESGAnalyzer(RegulatoryNLP):
    """Specialized analyzer for supplier ESG documents"""
    
    def __init__(self):
        super().__init__()
        
        # Additional patterns for supplier ESG documents
        self.esg_patterns = {
            'SCOPE1_VALUE': [
                r'\b(scope\s*1|direct\s+emissions?)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(tons?|kg)\s*from\s*(scope\s*1|direct)\b'
            ],
            'SCOPE2_VALUE': [
                r'\b(scope\s*2|indirect\s+energy\s+emissions?)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(tons?|kg)\s*from\s*(scope\s*2|purchased\s+electricity)\b'
            ],
            'SCOPE3_VALUE': [
                r'\b(scope\s*3|value\s+chain\s+emissions?)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(tons?|kg)\s*from\s*(scope\s*3|supply\s+chain)\b'
            ],
            'TOTAL_EMISSIONS': [
                r'\b(total\s+emissions?|overall\s+emissions?|emissions?\s+total)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(tons?|kg)\s*CO2e\s*(?:in\s+total|overall)\b'
            ],
            'CERTIFICATION': [
                r'\b(ISO\s*14001|ISO\s*14064|SBTi|CDP\s+[A-F]|EcoVadis\s+(?:gold|silver|bronze))\b',
                r'\b(certified|certification)\s+(?:to|for)\s+(ISO\s*14001|carbon\s+neutral)\b'
            ]
        }
    
    def analyze_supplier_esg(self, text, supplier_id=None):
        """Analyze supplier ESG documents"""
        
        # Run standard regulatory analysis
        base_analysis = self.extract_regulatory_entities(text, use_model=False)
        
        # Extract ESG-specific entities
        esg_entities = self._extract_esg_entities(text)
        
        # Calculate data quality score
        quality_score = self._calculate_data_quality(text, esg_entities)
        
        # Identify reporting year
        reporting_year = self._extract_reporting_year(text)
        
        return {
            'success': True,
            'supplier_id': supplier_id,
            'metadata': {
                **base_analysis['metadata'],
                'data_quality_score': quality_score,
                'reporting_year': reporting_year
            },
            'emission_data': {
                'scope1': self._extract_scope_value(text, 'scope1'),
                'scope2': self._extract_scope_value(text, 'scope2'),
                'scope3': self._extract_scope_value(text, 'scope3'),
                'total': self._extract_total_emission(text)
            },
            'renewable_energy_pct': self._extract_renewable_pct(text),
            'certifications': self._extract_certifications(text),
            'completeness_check': self._check_completeness(text),
            'esg_entities': esg_entities
        }
    
    def _extract_esg_entities(self, text):
        """Extract ESG-specific entities"""
        esg_entities = {}
        
        for entity_type, patterns in self.esg_patterns.items():
            esg_entities[entity_type] = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # Get the value (usually last element)
                        value = match[-2] if len(match) >= 2 else match[0]
                        esg_entities[entity_type].append(value)
                    else:
                        esg_entities[entity_type].append(match)
        
        return esg_entities
    
    def _extract_scope_value(self, text, scope):
        """Extract specific scope emission value"""
        patterns = {
            'scope1': [
                r'\b(scope\s*1|direct\s+emissions?)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg)\s*(?:from\s+)?scope\s*1\b'
            ],
            'scope2': [
                r'\b(scope\s*2|indirect\s+energy)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg)\s*(?:from\s+)?scope\s*2\b'
            ],
            'scope3': [
                r'\b(scope\s*3|value\s+chain)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg|tCO2e?)\b',
                r'\b(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg)\s*(?:from\s+)?scope\s*3\b'
            ]
        }
        
        for pattern in patterns.get(scope, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Try to extract numeric value
                for match in matches:
                    if isinstance(match, tuple):
                        for item in match:
                            if re.match(r'\d', str(item)):
                                try:
                                    # Clean and convert to number
                                    value_str = str(item).replace(',', '')
                                    return int(float(value_str))
                                except:
                                    continue
                    else:
                        if re.match(r'\d', str(match)):
                            try:
                                value_str = str(match).replace(',', '')
                                return int(float(value_str))
                            except:
                                continue
        
        return None
    
    def _extract_total_emission(self, text):
        """Extract total emission value"""
        patterns = [
            r'\b(total\s+emissions?|overall\s+emissions?)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg|tCO2e?)\b',
            r'\b(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg)\s*CO2e\s*(?:in\s+total|overall|total)\b',
            r'\b(emitted|released)\s+(\d{1,3}(?:,\d{3})*)\s*(?:tons?|kg)\s*CO2\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        for item in match:
                            if re.match(r'\d', str(item)):
                                try:
                                    value_str = str(item).replace(',', '')
                                    return int(float(value_str))
                                except:
                                    continue
        
        return None
    
    def _extract_renewable_pct(self, text):
        """Extract renewable energy percentage"""
        patterns = [
            r'\b(\d{1,3})%\s*(?:renewable|green|clean)\s*(?:energy|electricity|power)\b',
            r'\b(?:renewable|green)\s*(?:energy|power)\s*(\d{1,3})%\b',
            r'\b(\d{1,3})%\s*(?:from\s+)?(?:renewable|solar|wind|hydro)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    try:
                        return int(match)
                    except:
                        continue
        
        return None
    
    def _extract_certifications(self, text):
        """Extract certifications mentioned"""
        certifications = []
        patterns = [
            r'\b(ISO\s*14001|ISO\s*14064|ISO\s*50001)\b',
            r'\b(SBTi|Science\s+Based\s+Targets)\b',
            r'\b(CDP\s+[A-F]|Carbon\s+Disclosure\s+Project)\b',
            r'\b(EcoVadis\s+(?:gold|silver|bronze|platinum))\b',
            r'\b(carbon\s+neutral|net\s+zero|climate\s+neutral)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        
        return list(set(certifications))
    
    def _calculate_data_quality(self, text, esg_entities):
        """Calculate data quality score (0-100)"""
        score = 50  # Base score
        
        # Check for scope emissions
        has_scope1 = bool(esg_entities.get('SCOPE1_VALUE'))
        has_scope2 = bool(esg_entities.get('SCOPE2_VALUE'))
        has_scope3 = bool(esg_entities.get('SCOPE3_VALUE'))
        
        if has_scope1:
            score += 10
        if has_scope2:
            score += 10
        if has_scope3:
            score += 15
        
        if esg_entities.get('TOTAL_EMISSIONS'):
            score += 10

        if self._extract_renewable_pct(text):
            score += 10
        
        if self._extract_certifications(text):
            score += 5
        
        return min(score, 100)
    
    def _extract_reporting_year(self, text):
        """Extract reporting year from text"""
        year_patterns = [
            r'\b(?:for|in|year|reporting)\s+(\d{4})\b',
            r'\b(\d{4})\s*(?:sustainability|ESG|emissions?)\s+report\b',
            r'\b(?:calendar|fiscal)\s+year\s+(\d{4})\b'
        ]
        
        current_year = datetime.now().year
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for year_str in matches:
                    try:
                        year = int(year_str)
                        if 2000 <= year <= current_year:
                            return year
                    except:
                        continue
        
        # Default to current year if not found
        return current_year
    
    def _check_completeness(self, text):
        """Check completeness of ESG disclosure"""
        checks = {
            'has_scope1': bool(self._extract_scope_value(text, 'scope1')),
            'has_scope2': bool(self._extract_scope_value(text, 'scope2')),
            'has_scope3': bool(self._extract_scope_value(text, 'scope3')),
            'has_total': bool(self._extract_total_emission(text)),
            'has_renewable_pct': bool(self._extract_renewable_pct(text)),
            'has_certifications': bool(self._extract_certifications(text)),
            'has_reporting_year': bool(self._extract_reporting_year(text) != datetime.now().year)
        }
        
        return checks

__all__ = ['RegulatoryNLP', 'RegulatoryNLPModel', 'SupplierESGAnalyzer']