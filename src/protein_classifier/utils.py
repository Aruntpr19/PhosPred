"""
Utility functions for protein binding site classification.
"""

import re
import json
import numpy as np
import torch
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Standard amino acids
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
EXTENDED_AA = STANDARD_AA | set('BJOUXZ')  # Include ambiguous/non-standard


def validate_sequence(sequence: str, allow_extended: bool = True) -> None:
    """
    Validate a protein sequence.
    
    Args:
        sequence: Protein sequence string
        allow_extended: Whether to allow extended amino acid alphabet
        
    Raises:
        ValueError: If sequence is invalid
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    
    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string")
    
    # Convert to uppercase and remove whitespace
    sequence = sequence.upper().strip()
    
    # Check for valid amino acids
    valid_aa = EXTENDED_AA if allow_extended else STANDARD_AA
    invalid_chars = set(sequence) - valid_aa
    
    if invalid_chars:
        raise ValueError(f"Invalid amino acid characters: {invalid_chars}")
    
    if len(sequence) > 1025:
        logger.warning(f"Sequence is very long ({len(sequence)} residues). This may cause memory issues.")


def validate_indices(indices: List[int], sequence_length: int) -> None:
    """
    Validate binding site indices.
    
    Args:
        indices: List of residue indices (1-based)
        sequence_length: Length of the protein sequence
        
    Raises:
        ValueError: If indices are invalid
    """
    if not indices:
        raise ValueError("Binding site indices cannot be empty")
    
    if not all(isinstance(idx, int) for idx in indices):
        raise ValueError("All binding site indices must be integers")
    
    if any(idx < 1 for idx in indices):
        raise ValueError("Binding site indices must be 1-based (>= 1)")
    
    if any(idx > sequence_length for idx in indices):
        raise ValueError(f"Some indices exceed sequence length ({sequence_length})")
    
    if len(set(indices)) != len(indices):
        logger.warning("Duplicate indices found in binding site list")


def clean_sequence(sequence: str) -> str:
    """
    Clean and normalize a protein sequence.
    
    Args:
        sequence: Raw protein sequence
        
    Returns:
        Cleaned sequence string
    """
    # Remove whitespace and convert to uppercase
    sequence = re.sub(r'\s+', '', sequence.upper())
    
    # Remove common non-amino acid characters
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBJOUXZ]', '', sequence)
    
    return sequence


def load_sequences_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load protein sequences and binding sites from a JSON file.
    
    Expected format:
    {
        "sequences": [
            {
                "id": "protein_1",
                "sequence": "MKVLWAALLVTFLAG...",
                "binding_sites": [23, 45, 67],
                "label": "phosphate"
            }
        ]
    }
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate data format
    if 'sequences' not in data:
        raise ValueError("JSON file must contain 'sequences' key")
    
    sequences = data['sequences']
    if not isinstance(sequences, list):
        raise ValueError("'sequences' must be a list")
    
    # Validate each sequence entry
    for i, seq_data in enumerate(sequences):
        if not isinstance(seq_data, dict):
            raise ValueError(f"Sequence {i} must be a dictionary")
        
        required_keys = ['sequence', 'binding_sites']
        for key in required_keys:
            if key not in seq_data:
                raise ValueError(f"Sequence {i} missing required key: {key}")
        
        # Clean sequence
        seq_data['sequence'] = clean_sequence(seq_data['sequence'])
        
        # Validate
        validate_sequence(seq_data['sequence'])
        validate_indices(seq_data['binding_sites'], len(seq_data['sequence']))
    
    return data


def save_predictions_to_file(predictions: List[Dict], 
                           file_path: Union[str, Path],
                           include_metadata: bool = True) -> None:
    """
    Save prediction results to a JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        file_path: Output file path
        include_metadata: Whether to include metadata
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'predictions': predictions
    }
    
    if include_metadata:
        output_data['metadata'] = {
            'num_predictions': len(predictions),
            'classes': ['phosphate', 'sulfate', 'chloride', 'nitrate', 'carbonate'],
            'format_version': '1.0'
        }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Predictions saved to {file_path}")


def calculate_metrics(y_true: List[str], y_pred: List[str], 
                     classes: List[str]) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted labels must match")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    
    # Overall metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    for i, class_name in enumerate(classes):
        metrics['per_class_metrics'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        }
    
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def format_prediction_output(results: Dict[str, float], 
                           top_k: int = 3,
                           threshold: float = 0.1) -> str:
    """
    Format prediction results for display.
    
    Args:
        results: Prediction probabilities dictionary
        top_k: Number of top predictions to show
        threshold: Minimum probability threshold
        
    Returns:
        Formatted string
    """
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    lines = []
    lines.append("Prediction Results:")
    lines.append("-" * 30)
    
    count = 0
    for class_name, probability in sorted_results:
        if count >= top_k and probability < threshold:
            break
        
        lines.append(f"{class_name:>12}: {probability:.4f} ({probability*100:.2f}%)")
        count += 1
    
    return "\n".join(lines)


def batch_process_sequences(sequences: List[str],
                          binding_sites_list: List[List[int]],
                          classifier,
                          batch_size: int = 8,
                          show_progress: bool = True) -> List[Optional[Dict[str, float]]]:
    """
    Process multiple sequences in batches.
    
    Args:
        sequences: List of protein sequences
        binding_sites_list: List of binding site indices for each sequence
        classifier: ProteinBindingSiteClassifier instance
        batch_size: Number of sequences to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        List of prediction results
    """
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(sequences), batch_size), desc="Processing batches")
        except ImportError:
            logger.warning("tqdm not available. Progress bar disabled.")
            iterator = range(0, len(sequences), batch_size)
    else:
        iterator = range(0, len(sequences), batch_size)
    
    results = []
    
    for i in iterator:
        batch_sequences = sequences[i:i + batch_size]
        batch_binding_sites = binding_sites_list[i:i + batch_size]
        
        for seq, sites in zip(batch_sequences, batch_binding_sites):
            try:
                result = classifier.predict(seq, sites)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sequence {len(results)}: {e}")
                results.append(None)
    
    return results


def create_example_data() -> Dict[str, Any]:
    """
    Create example data for testing and demonstration.
    
    Returns:
        Dictionary with example sequences and binding sites
    """
    examples = {
        "sequences": [
            {
                "id": "example_1",
                "sequence": "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQZALLDALQHIAAQLFLMAGILQNMHLDILKQVLALRLAGVMQRLRLRYQSGILGVQLLVSQVLPDLDHYQNLIRILIQKQLL",
                "binding_sites": [23, 45, 67, 89, 102],
                "label": "phosphate",
                "description": "ATP binding site example"
            },
            {
                "id": "example_2", 
                "sequence": "MTMDKSELVQKAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIREYRQMVETELKLICCDILDVLDKHLIPAANTGESKVFYYKMKGDYHRYLAEFATGNDRKEAAMELNYIPNRVAQQLAGKQSLLIGVATSSLALHAPSQIVAVRQPYDRLDVGDIFEAQKIEWHE",
                "binding_sites": [12, 34, 56, 78, 91, 134],
                "label": "sulfate",
                "description": "Sulfate transport protein example"
            },
            {
                "id": "example_3",
                "sequence": "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLAGSGSGHMHHHHH",
                "binding_sites": [45, 67, 89, 92, 95],
                "label": "chloride",
                "description": "Chloride channel example"
            }
        ],
        "metadata": {
            "description": "Example protein sequences for binding site classification",
            "classes": ["phosphate", "sulfate", "chloride", "nitrate", "carbonate"],
            "created_for": "demonstration_and_testing"
        }
    }
    
    return examples


def estimate_memory_usage(sequence_length: int, batch_size: int = 1) -> Dict[str, float]:
    """
    Estimate memory usage for processing sequences.
    
    Args:
        sequence_length: Length of the protein sequence
        batch_size: Number of sequences to process together
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # ESM2 embedding dimension (for esm2_t6_8M_UR50D)
    embedding_dim = 320
    
    # Estimates in bytes
    esm2_embeddings = sequence_length * embedding_dim * 4 * batch_size  # float32
    esm2_model_params = 8_000_000 * 4  # Approximate ESM2 parameters
    prediction_model = 1_000_000 * 4  # Approximate prediction model size
    
    # Convert to MB
    estimates = {
        'esm2_embeddings_mb': esm2_embeddings / (1024 * 1024),
        'esm2_model_mb': esm2_model_params / (1024 * 1024),
        'prediction_model_mb': prediction_model / (1024 * 1024),
        'total_estimated_mb': (esm2_embeddings + esm2_model_params + prediction_model) / (1024 * 1024)
    }
    
    return estimates