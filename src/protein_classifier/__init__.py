"""
Protein Binding Site Classifier

A deep learning tool for predicting protein binding site types using ESM2 embeddings.
"""

__version__ = "1.0.0"
__author__ = "Arunraj Balasubramanian"
__email__ = "arunrajtpr19@gmail.com"

from .classifier import ProteinBindingSiteClassifier
from .model import PredictionModel, AttentionPredictionModel
from .utils import (
    validate_sequence,
    validate_indices,
    clean_sequence,
    load_sequences_from_file,
    save_predictions_to_file,
    calculate_metrics,
    format_prediction_output,
    batch_process_sequences,
    create_example_data,
    estimate_memory_usage
)

__all__ = [
    "ProteinBindingSiteClassifier",
    "PredictionModel", 
    "AttentionPredictionModel",
    "validate_sequence",
    "validate_indices", 
    "clean_sequence",
    "load_sequences_from_file",
    "save_predictions_to_file",
    "calculate_metrics",
    "format_prediction_output",
    "batch_process_sequences",
    "create_example_data",
    "estimate_memory_usage"
]

# Package metadata
__title__ = "protein-binding-classifier"
__description__ = "A deep learning tool for predicting protein binding site types using ESM2 embeddings"
__url__ = "https://github.com/yourusername/protein-binding-classifier"
__license__ = "MIT"
__copyright__ = "Copyright 2025"