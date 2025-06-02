"""
Advanced metrics for medical image report generation and other tasks
"""

import numpy as np
from collections import Counter
import re
from typing import List, Dict, Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)


class GREENScorer:
    """
    GREEN Score implementation for medical report evaluation.
    
    GREEN (Grounding Radiology Report Evaluation with Natural Language) Score
    evaluates the clinical accuracy and completeness of generated medical reports.
    
    This version includes:
    1. Clinical entity matching with severity
    2. Anatomical location grounding with laterality
    3. Temporal information handling
    4. Size and measurement accuracy
    5. Uncertainty and negation handling
    6. Clinical significance weighting
    7. Comparison with prior studies
    8. Structured reporting elements
    """
    
    def __init__(self):
        # Common medical entities and their variations with severity weights
        self.clinical_entities = {
            'normal': {
                'terms': ['normal', 'unremarkable', 'no abnormality', 'clear', 'no acute findings'],
                'severity': 0,
                'weight': 1.0
            },
            'opacity': {
                'terms': ['opacity', 'opacities', 'opacification', 'density', 'densities'],
                'severity': 2,
                'weight': 0.8
            },
            'consolidation': {
                'terms': ['consolidation', 'consolidations', 'consolidated', 'airspace disease'],
                'severity': 3,
                'weight': 0.9
            },
            'effusion': {
                'terms': ['effusion', 'effusions', 'fluid', 'pleural fluid'],
                'severity': 2,
                'weight': 0.8
            },
            'pneumothorax': {
                'terms': ['pneumothorax', 'pneumothoraces', 'collapsed lung', 'air leak'],
                'severity': 4,
                'weight': 1.0
            },
            'nodule': {
                'terms': ['nodule', 'nodules', 'mass', 'lesion', 'tumor', 'neoplasm'],
                'severity': 3,
                'weight': 0.9
            },
            'fracture': {
                'terms': ['fracture', 'fractures', 'fractured', 'break', 'broken'],
                'severity': 3,
                'weight': 0.9
            },
            'cardiomegaly': {
                'terms': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'heart enlargement'],
                'severity': 2,
                'weight': 0.7
            },
            'edema': {
                'terms': ['edema', 'edematous', 'swelling', 'pulmonary edema', 'interstitial edema'],
                'severity': 2,
                'weight': 0.8
            },
            'infiltrate': {
                'terms': ['infiltrate', 'infiltrates', 'infiltration', 'interstitial markings'],
                'severity': 2,
                'weight': 0.7
            },
            'atelectasis': {
                'terms': ['atelectasis', 'atelectatic', 'collapse', 'volume loss'],
                'severity': 2,
                'weight': 0.7
            },
            'pneumonia': {
                'terms': ['pneumonia', 'pneumonic', 'infection', 'infectious process'],
                'severity': 3,
                'weight': 0.9
            },
        }
        
        # Anatomical locations with laterality
        self.anatomical_locations = {
            'lung': ['lung', 'lungs', 'pulmonary', 'pulmonic'],
            'lobe': ['lobe', 'lobar'],
            'left': ['left', 'left-sided', 'left side'],
            'right': ['right', 'right-sided', 'right side'],
            'upper': ['upper', 'superior', 'apex', 'apical', 'apices'],
            'lower': ['lower', 'inferior', 'base', 'basal', 'basilar'],
            'middle': ['middle', 'mid', 'central', 'lingula', 'lingular'],
            'bilateral': ['bilateral', 'both', 'bibasilar', 'bilaterally'],
            'unilateral': ['unilateral', 'one side', 'single'],
            'heart': ['heart', 'cardiac', 'cardiovascular', 'pericardium'],
            'mediastinum': ['mediastinum', 'mediastinal', 'hilar', 'hilum'],
            'pleura': ['pleura', 'pleural', 'pleural space'],
            'chest': ['chest', 'thorax', 'thoracic', 'hemithorax'],
            'diaphragm': ['diaphragm', 'diaphragmatic', 'hemidiaphragm'],
            'costophrenic': ['costophrenic', 'cp angle', 'costophrenic angle'],
        }
        
        # Severity modifiers
        self.severity_modifiers = {
            'mild': 0.3,
            'minimal': 0.2,
            'slight': 0.2,
            'moderate': 0.6,
            'marked': 0.8,
            'severe': 1.0,
            'extensive': 0.9,
            'large': 0.8,
            'small': 0.3,
            'tiny': 0.1,
            'trace': 0.1,
            'prominent': 0.7,
            'significant': 0.8
        }
        
        # Temporal indicators
        self.temporal_indicators = {
            'new': ['new', 'newly', 'recent', 'acute'],
            'old': ['old', 'chronic', 'longstanding', 'prior'],
            'unchanged': ['unchanged', 'stable', 'no change', 'similar'],
            'improved': ['improved', 'improving', 'decreased', 'resolving'],
            'worsened': ['worsened', 'worsening', 'increased', 'progressed']
        }
        
        # Uncertainty markers
        self.uncertainty_markers = {
            'possible': 0.3,
            'probable': 0.7,
            'likely': 0.7,
            'unlikely': 0.2,
            'questionable': 0.3,
            'suspected': 0.6,
            'concerning for': 0.7,
            'suggestive of': 0.6,
            'may represent': 0.5,
            'cannot exclude': 0.4,
            'differential includes': 0.5
        }
        
        # Size patterns
        self.size_patterns = [
            r'(\d+\.?\d*)\s*(cm|mm|millimeter|centimeter)',
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(cm|mm)',
            r'measuring\s*(\d+\.?\d*)\s*(cm|mm)',
            r'size\s*of\s*(\d+\.?\d*)\s*(cm|mm)'
        ]
        
        # Negation patterns
        self.negation_patterns = [
            r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\babsent\b',
            r'\bnegative\b', r'\bdenied\b', r'\bnone\b',
            r'\bfree of\b', r'\brule out\b', r'\bruled out\b',
            r'\bno evidence\b', r'\bno signs\b'
        ]
        
        # Clinical significance weights for different finding types
        self.clinical_significance = {
            'critical': 1.0,    # Life-threatening findings
            'urgent': 0.9,      # Requires urgent attention
            'important': 0.7,   # Clinically significant
            'minor': 0.4,       # Minor findings
            'incidental': 0.2   # Incidental findings
        }
        
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract clinical entities with severity and context from text"""
        text_lower = text.lower()
        entities = {}
        
        # Extract clinical entities
        for entity_type, entity_info in self.clinical_entities.items():
            matches = []
            for term in entity_info['terms']:
                for match in re.finditer(r'\b' + re.escape(term) + r'\b', text_lower):
                    # Extract context around the entity
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text_lower), match.end() + 50)
                    context = text_lower[context_start:context_end]
                    
                    # Determine severity
                    severity = self._extract_severity(context, entity_info['severity'])
                    
                    # Check for uncertainty
                    uncertainty = self._extract_uncertainty(context)
                    
                    # Extract size if applicable
                    size = self._extract_size(context)
                    
                    matches.append({
                        'term': term,
                        'start': match.start(),
                        'end': match.end(),
                        'severity': severity,
                        'uncertainty': uncertainty,
                        'size': size,
                        'weight': entity_info['weight']
                    })
            
            if matches:
                entities[entity_type] = matches
                
        return entities
    
    def _extract_severity(self, context: str, base_severity: float) -> float:
        """Extract severity modifier from context"""
        for modifier, weight in self.severity_modifiers.items():
            if modifier in context:
                return base_severity * weight
        return base_severity
    
    def _extract_uncertainty(self, context: str) -> float:
        """Extract uncertainty level from context"""
        for marker, confidence in self.uncertainty_markers.items():
            if marker in context:
                return 1.0 - confidence  # Convert confidence to uncertainty
        return 0.0  # No uncertainty
    
    def _extract_size(self, context: str) -> Optional[str]:
        """Extract size measurements from context"""
        for pattern in self.size_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(0)
        return None
    
    def extract_locations(self, text: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """Extract anatomical locations from text"""
        text_lower = text.lower()
        locations = {}
        
        for location_type, variations in self.anatomical_locations.items():
            matches = []
            for variation in variations:
                for match in re.finditer(r'\b' + re.escape(variation) + r'\b', text_lower):
                    matches.append((variation, match.start(), match.end()))
            if matches:
                locations[location_type] = matches
                
        return locations
    
    def check_negation(self, text: str, entity_position: int) -> bool:
        """Check if an entity is negated in the text"""
        text_lower = text.lower()
        
        # Look for negation within a window before the entity
        window_start = max(0, entity_position - 50)
        window_text = text_lower[window_start:entity_position]
        
        for pattern in self.negation_patterns:
            if re.search(pattern, window_text):
                return True
        
        return False
    
    def calculate_green_score(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate GREEN score components
        
        Returns:
            dict: Contains overall score and detailed component scores
        """
        # Extract information from both texts
        gen_entities = self.extract_entities(generated)
        ref_entities = self.extract_entities(reference)
        gen_locations = self.extract_locations(generated)
        ref_locations = self.extract_locations(reference)
        gen_temporal = self._extract_temporal_info(generated)
        ref_temporal = self._extract_temporal_info(reference)
        
        # Component scores with enhanced evaluation
        
        # 1. Entity matching with severity and uncertainty (30%)
        entity_score = self._calculate_entity_matching_score(
            generated, reference, gen_entities, ref_entities
        )
        
        # 2. Location accuracy with laterality (20%)
        location_score = self._calculate_location_score(
            gen_locations, ref_locations
        )
        
        # 3. Negation and uncertainty handling (15%)
        negation_score = self._calculate_negation_score(
            generated, reference, gen_entities, ref_entities
        )
        
        # 4. Temporal accuracy (10%)
        temporal_score = self._calculate_temporal_score(
            gen_temporal, ref_temporal, gen_entities, ref_entities
        )
        
        # 5. Size/measurement accuracy (10%)
        measurement_score = self._calculate_measurement_score(
            gen_entities, ref_entities
        )
        
        # 6. Clinical significance alignment (10%)
        significance_score = self._calculate_clinical_significance_score(
            gen_entities, ref_entities
        )
        
        # 7. Report structure and completeness (5%)
        structure_score = self._calculate_structure_score(
            generated, reference
        )
        
        # Calculate overall GREEN score with weighted components
        overall_score = (
            0.30 * entity_score +
            0.20 * location_score +
            0.15 * negation_score +
            0.10 * temporal_score +
            0.10 * measurement_score +
            0.10 * significance_score +
            0.05 * structure_score
        )
        
        return {
            'overall': overall_score,
            'entity_matching': entity_score,
            'location_accuracy': location_score,
            'negation_handling': negation_score,
            'temporal_accuracy': temporal_score,
            'measurement_accuracy': measurement_score,
            'clinical_significance': significance_score,
            'structure_completeness': structure_score,
            # Additional detailed metrics
            'entity_details': self._get_entity_details(gen_entities, ref_entities),
            'severity_correlation': self._calculate_severity_correlation(gen_entities, ref_entities)
        }
    
    def _calculate_entity_matching_score(self, gen_text: str, ref_text: str,
                                       gen_entities: Dict, ref_entities: Dict) -> float:
        """Calculate entity matching score with severity and uncertainty"""
        if not ref_entities:
            return 1.0 if not gen_entities else 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for entity_type, ref_mentions in ref_entities.items():
            if entity_type in gen_entities:
                gen_mentions = gen_entities[entity_type]
                
                # For each reference mention, find best matching generated mention
                for ref_mention in ref_mentions:
                    ref_negated = self.check_negation(ref_text, ref_mention['start'])
                    
                    best_match_score = 0.0
                    for gen_mention in gen_mentions:
                        gen_negated = self.check_negation(gen_text, gen_mention['start'])
                        
                        # Calculate match score considering multiple factors
                        match_score = 0.0
                        
                        # 1. Negation match (40%)
                        if ref_negated == gen_negated:
                            match_score += 0.4
                        
                        # 2. Severity match (30%)
                        severity_diff = abs(ref_mention['severity'] - gen_mention['severity'])
                        severity_score = max(0, 1 - severity_diff / 4.0)  # Normalize by max severity
                        match_score += 0.3 * severity_score
                        
                        # 3. Uncertainty match (20%)
                        uncertainty_diff = abs(ref_mention['uncertainty'] - gen_mention['uncertainty'])
                        uncertainty_score = max(0, 1 - uncertainty_diff)
                        match_score += 0.2 * uncertainty_score
                        
                        # 4. Size match if applicable (10%)
                        if ref_mention['size'] and gen_mention['size']:
                            size_score = self._compare_sizes(ref_mention['size'], gen_mention['size'])
                            match_score += 0.1 * size_score
                        elif not ref_mention['size'] and not gen_mention['size']:
                            match_score += 0.1
                        
                        best_match_score = max(best_match_score, match_score)
                    
                    # Weight by clinical importance
                    weight = ref_mention['weight']
                    total_score += best_match_score * weight
                    total_weight += weight
            else:
                # Penalize missing entity types based on their clinical weight
                for ref_mention in ref_mentions:
                    total_weight += ref_mention['weight']
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_location_score(self, gen_locations: Dict, ref_locations: Dict) -> float:
        """Calculate anatomical location accuracy score"""
        if not ref_locations:
            return 1.0 if not gen_locations else 0.0
        
        ref_location_set = set(ref_locations.keys())
        gen_location_set = set(gen_locations.keys())
        
        if not ref_location_set:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = ref_location_set & gen_location_set
        union = ref_location_set | gen_location_set
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_negation_score(self, gen_text: str, ref_text: str,
                                 gen_entities: Dict, ref_entities: Dict) -> float:
        """Calculate negation and uncertainty handling accuracy"""
        total_score = 0.0
        total_count = 0
        
        # Check negation and uncertainty handling for matching entities
        for entity_type in set(gen_entities.keys()) & set(ref_entities.keys()):
            for ref_mention in ref_entities[entity_type]:
                ref_negated = self.check_negation(ref_text, ref_mention['start'])
                ref_uncertainty = ref_mention['uncertainty']
                
                best_score = 0.0
                # Find best matching mention in generated text
                for gen_mention in gen_entities[entity_type]:
                    gen_negated = self.check_negation(gen_text, gen_mention['start'])
                    gen_uncertainty = gen_mention['uncertainty']
                    
                    # Score based on both negation and uncertainty match
                    score = 0.0
                    if ref_negated == gen_negated:
                        score += 0.6  # 60% for correct negation
                    
                    # Uncertainty match (within tolerance)
                    uncertainty_diff = abs(ref_uncertainty - gen_uncertainty)
                    uncertainty_score = max(0, 1 - uncertainty_diff * 2)  # Tolerance of 0.5
                    score += 0.4 * uncertainty_score
                    
                    best_score = max(best_score, score)
                
                total_score += best_score
                total_count += 1
        
        return total_score / total_count if total_count > 0 else 1.0
    
    def _extract_temporal_info(self, text: str) -> Dict[str, List[int]]:
        """Extract temporal information from text"""
        text_lower = text.lower()
        temporal_info = {}
        
        for temp_type, indicators in self.temporal_indicators.items():
            positions = []
            for indicator in indicators:
                for match in re.finditer(r'\b' + re.escape(indicator) + r'\b', text_lower):
                    positions.append(match.start())
            if positions:
                temporal_info[temp_type] = positions
        
        return temporal_info
    
    def _calculate_temporal_score(self, gen_temporal: Dict, ref_temporal: Dict, 
                                gen_entities: Dict, ref_entities: Dict) -> float:
        """Calculate temporal information accuracy"""
        if not ref_temporal:
            return 1.0 if not gen_temporal else 0.5
        
        # Check temporal consistency for entities
        score = 0.0
        total = 0
        
        for temp_type in ref_temporal:
            if temp_type in gen_temporal:
                score += 1.0
            total += 1
        
        # Also check if temporal info is correctly associated with entities
        temporal_entity_score = self._check_temporal_entity_association(
            gen_temporal, ref_temporal, gen_entities, ref_entities
        )
        
        return (score / total * 0.5 + temporal_entity_score * 0.5) if total > 0 else 0.5
    
    def _check_temporal_entity_association(self, gen_temporal: Dict, ref_temporal: Dict,
                                         gen_entities: Dict, ref_entities: Dict) -> float:
        """Check if temporal information is correctly associated with entities"""
        # Simplified check - in production, would do proximity analysis
        if gen_temporal and gen_entities and ref_temporal and ref_entities:
            return 0.8  # Placeholder - would implement proximity matching
        return 0.5
    
    def _calculate_measurement_score(self, gen_entities: Dict, ref_entities: Dict) -> float:
        """Calculate size/measurement accuracy"""
        total_score = 0.0
        total_count = 0
        
        for entity_type in ref_entities:
            if entity_type in gen_entities:
                for ref_mention in ref_entities[entity_type]:
                    if ref_mention['size']:
                        best_score = 0.0
                        for gen_mention in gen_entities[entity_type]:
                            if gen_mention['size']:
                                score = self._compare_sizes(ref_mention['size'], gen_mention['size'])
                                best_score = max(best_score, score)
                        
                        total_score += best_score
                        total_count += 1
        
        return total_score / total_count if total_count > 0 else 1.0
    
    def _compare_sizes(self, size1: str, size2: str) -> float:
        """Compare two size measurements"""
        # Extract numeric values
        nums1 = re.findall(r'(\d+\.?\d*)', size1)
        nums2 = re.findall(r'(\d+\.?\d*)', size2)
        
        if not nums1 or not nums2:
            return 0.0
        
        # Simple comparison - in production would handle units and ranges
        try:
            val1 = float(nums1[0])
            val2 = float(nums2[0])
            
            # Calculate relative difference
            if val1 == 0 and val2 == 0:
                return 1.0
            
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return 1.0
                
            diff = abs(val1 - val2) / max_val
            
            # Score based on difference
            if diff < 0.1:  # Within 10%
                return 1.0
            elif diff < 0.2:  # Within 20%
                return 0.8
            elif diff < 0.5:  # Within 50%
                return 0.5
            else:
                return 0.2
        except (ValueError, IndexError):
            return 0.0
    
    def _calculate_clinical_significance_score(self, gen_entities: Dict, ref_entities: Dict) -> float:
        """Calculate how well clinical significance is captured"""
        # Prioritize high-severity findings
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for entity_type in ref_entities:
            entity_weight = self.clinical_entities[entity_type]['weight']
            
            if entity_type in gen_entities:
                # Check if high-severity findings are captured
                for ref_mention in ref_entities[entity_type]:
                    severity = ref_mention['severity']
                    weight = entity_weight * (1 + severity / 4.0)  # Higher weight for severe findings
                    
                    # Check if captured in generated
                    captured = any(
                        self._entities_match(ref_mention, gen_mention) 
                        for gen_mention in gen_entities[entity_type]
                    )
                    
                    if captured:
                        total_weighted_score += weight
                    total_weight += weight
            else:
                # Missing entity type - weight by severity
                for ref_mention in ref_entities[entity_type]:
                    severity = ref_mention['severity']
                    weight = entity_weight * (1 + severity / 4.0)
                    total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _entities_match(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entity mentions match"""
        # Simple term match - in production would be more sophisticated
        return entity1['term'] == entity2['term']
    
    def _calculate_structure_score(self, generated: str, reference: str) -> float:
        """Evaluate report structure and completeness"""
        # Check for key sections
        sections = ['findings', 'impression', 'comparison', 'technique', 'recommendation']
        
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        
        score = 0.0
        for section in sections:
            if section in ref_lower:
                if section in gen_lower:
                    score += 1.0
        
        # Check relative length
        len_ratio = len(generated) / len(reference) if len(reference) > 0 else 0
        length_score = min(1.0, len_ratio) if len_ratio < 2.0 else max(0, 2.0 - len_ratio)
        
        # Combine section presence and length appropriateness
        total_sections = sum(1 for s in sections if s in ref_lower)
        section_score = score / total_sections if total_sections > 0 else 1.0
        
        return section_score * 0.7 + length_score * 0.3
    
    def _get_entity_details(self, gen_entities: Dict, ref_entities: Dict) -> Dict:
        """Get detailed entity comparison statistics"""
        details = {
            'total_ref_entities': sum(len(e) for e in ref_entities.values()),
            'total_gen_entities': sum(len(e) for e in gen_entities.values()),
            'matched_entity_types': len(set(gen_entities.keys()) & set(ref_entities.keys())),
            'missed_entity_types': list(set(ref_entities.keys()) - set(gen_entities.keys())),
            'extra_entity_types': list(set(gen_entities.keys()) - set(ref_entities.keys()))
        }
        return details
    
    def _calculate_severity_correlation(self, gen_entities: Dict, ref_entities: Dict) -> float:
        """Calculate correlation between severity assessments"""
        ref_severities = []
        gen_severities = []
        
        for entity_type in set(gen_entities.keys()) & set(ref_entities.keys()):
            for ref_mention in ref_entities[entity_type]:
                # Find best matching generated mention
                for gen_mention in gen_entities[entity_type]:
                    if self._entities_match(ref_mention, gen_mention):
                        ref_severities.append(ref_mention['severity'])
                        gen_severities.append(gen_mention['severity'])
                        break
        
        if not ref_severities:
            return 0.0
        
        # Calculate Pearson correlation
        try:
            correlation = np.corrcoef(ref_severities, gen_severities)[0, 1]
            # Convert to 0-1 scale
            return (correlation + 1) / 2 if not np.isnan(correlation) else 0.5
        except:
            return 0.5


def calculate_green_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate GREEN score for a list of generated reports
    
    Args:
        predictions: List of generated reports
        references: List of reference reports
        
    Returns:
        dict: Contains detailed mean scores, standard deviations, and component breakdowns
    """
    scorer = GREENScorer()
    
    # Initialize score containers for all components
    all_scores = {
        'overall': [],
        'entity_matching': [],
        'location_accuracy': [],
        'negation_handling': [],
        'temporal_accuracy': [],
        'measurement_accuracy': [],
        'clinical_significance': [],
        'structure_completeness': [],
        'severity_correlation': []
    }
    
    # Additional detail tracking
    entity_details_list = []
    
    for pred, ref in zip(predictions, references):
        try:
            scores = scorer.calculate_green_score(pred, ref)
            
            # Extract main scores
            for key in all_scores:
                if key in scores:
                    value = scores[key]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        all_scores[key].append(value)
            
            # Track entity details separately
            if 'entity_details' in scores:
                entity_details_list.append(scores['entity_details'])
                
        except Exception as e:
            logger.warning(f"Error calculating GREEN score for a sample: {e}")
            # Add default scores for failed sample
            for key in all_scores:
                all_scores[key].append(0.0)
    
    # Calculate statistics for all components
    results = {}
    for key, values in all_scores.items():
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
            results[f'{key}_min'] = np.min(values)
            results[f'{key}_max'] = np.max(values)
        else:
            results[f'{key}_mean'] = 0.0
            results[f'{key}_std'] = 0.0
            results[f'{key}_min'] = 0.0
            results[f'{key}_max'] = 0.0
    
    # Add aggregate entity statistics
    if entity_details_list:
        results['avg_ref_entities'] = np.mean([d['total_ref_entities'] for d in entity_details_list])
        results['avg_gen_entities'] = np.mean([d['total_gen_entities'] for d in entity_details_list])
        results['avg_matched_types'] = np.mean([d['matched_entity_types'] for d in entity_details_list])
        
        # Collect all missed entity types
        all_missed = []
        for d in entity_details_list:
            all_missed.extend(d['missed_entity_types'])
        
        if all_missed:
            from collections import Counter
            missed_counter = Counter(all_missed)
            results['most_missed_entities'] = dict(missed_counter.most_common(5))
    
    # Add interpretation guidelines
    results['interpretation'] = {
        'overall': 'GREEN score (0-1, higher is better)',
        'entity_matching': 'Clinical finding identification accuracy',
        'location_accuracy': 'Anatomical location grounding precision',
        'temporal_accuracy': 'Temporal information consistency',
        'measurement_accuracy': 'Size/measurement precision',
        'clinical_significance': 'Critical finding prioritization',
        'structure_completeness': 'Report structure and completeness'
    }
    
    return results


def calculate_bleu_score(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """
    Calculate BLEU score for generated text
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        n_gram: Maximum n-gram to consider
        
    Returns:
        float: BLEU score
    """
    from collections import Counter
    import math
    
    def get_ngrams(text: str, n: int) -> Counter:
        """Extract n-grams from text"""
        tokens = text.lower().split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def modified_precision(prediction: str, reference: str, n: int) -> float:
        """Calculate modified precision for n-grams"""
        pred_ngrams = get_ngrams(prediction, n)
        ref_ngrams = get_ngrams(reference, n)
        
        if not pred_ngrams:
            return 0.0
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        
        return overlap / total if total > 0 else 0.0
    
    # Calculate brevity penalty
    total_pred_length = sum(len(p.split()) for p in predictions)
    total_ref_length = sum(len(r.split()) for r in references)
    
    if total_pred_length > total_ref_length:
        brevity_penalty = 1.0
    elif total_pred_length == 0:
        brevity_penalty = 0.0
    else:
        brevity_penalty = math.exp(1 - total_ref_length / total_pred_length)
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(n_gram + 1, 5)):
        precision_scores = []
        for pred, ref in zip(predictions, references):
            precision_scores.append(modified_precision(pred, ref, n))
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        if avg_precision > 0:
            precisions.append(math.log(avg_precision))
        else:
            # If any precision is 0, BLEU is 0
            return 0.0
    
    # Calculate BLEU score
    if not precisions:
        return 0.0
    
    bleu = brevity_penalty * math.exp(sum(precisions) / len(precisions))
    
    return bleu


def calculate_clinical_efficacy_score(predictions: List[str], references: List[str]) -> float:
    """
    Calculate clinical efficacy (CE) score for medical reports
    
    This measures how well the generated report captures clinically relevant findings
    """
    # Define clinically significant terms
    clinical_terms = {
        'critical': ['emergency', 'urgent', 'critical', 'acute', 'severe'],
        'abnormal': ['abnormal', 'abnormality', 'pathology', 'disease', 'disorder'],
        'normal': ['normal', 'unremarkable', 'clear', 'no acute', 'negative'],
        'uncertain': ['possible', 'probable', 'suspicious', 'concerning', 'question']
    }
    
    def categorize_report(text: str) -> str:
        """Categorize report as critical/abnormal/normal/uncertain"""
        text_lower = text.lower()
        
        # Check in order of priority
        for category, terms in clinical_terms.items():
            for term in terms:
                if term in text_lower:
                    # Check if negated
                    position = text_lower.find(term)
                    window = text_lower[max(0, position-20):position]
                    if not any(neg in window for neg in ['no ', 'not ', 'without']):
                        return category
        
        return 'normal'  # Default
    
    correct_categorizations = 0
    for pred, ref in zip(predictions, references):
        pred_category = categorize_report(pred)
        ref_category = categorize_report(ref)
        
        if pred_category == ref_category:
            correct_categorizations += 1
        elif (pred_category in ['critical', 'abnormal'] and 
              ref_category in ['critical', 'abnormal']):
            # Partial credit for getting severity approximately right
            correct_categorizations += 0.5
    
    return correct_categorizations / len(predictions) if predictions else 0.0 