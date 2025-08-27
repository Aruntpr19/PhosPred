"""
Protein Binding Site Classifier

A deep learning tool for predicting protein binding site types using ESM2 embeddings.
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import json

from .model import PredictionModel
from .utils import validate_sequence, validate_indices

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinBindingSiteClassifier:
    """
    A classifier for predicting protein binding site types using ESM2 embeddings.
    
    This class provides functionality to:
    - Generate ESM2 embeddings for protein sequences
    - Average embeddings at binding site positions
    - Predict binding site types using a neural network
    
    Attributes:
        classes (List[str]): List of binding site class names
        device (torch.device): Computing device (CPU/GPU)
        esm_model: ESM2 model for generating embeddings
        prediction_model: Neural network for classification
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
                 device: Optional[str] = None):
        """
        Initialize the protein binding site classifier.
        
        Args:
            model_path: Path to the trained prediction model (optional)
            esm_model_name: Name of the ESM model to use
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Define class labels
        self.classes = ['phosphate', 'sulfate', 'chloride', 'nitrate', 'carbonate']
        self.num_classes = len(self.classes)
        
        # Load ESM2 model and tokenizer
        logger.info(f"Loading ESM2 model: {esm_model_name}")
        try:
            self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
            self.esm_model = EsmModel.from_pretrained(esm_model_name)
            self.esm_model.to(self.device)
            self.esm_model.eval()
            
            # Get embedding dimension from model config
            self.embedding_dim = self.esm_model.config.hidden_size
            logger.info(f"ESM2 embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading ESM2 model: {e}")
            raise
        
        # Initialize prediction model
        self.prediction_model = PredictionModel(
            input_dim=self.embedding_dim,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Load trained model if provided
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No trained model loaded. Using random weights for demonstration.")
    
    def get_esm2_embeddings(self, sequence: str) -> torch.Tensor:
        """
        Generate ESM2 embeddings for a protein sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tensor of shape (sequence_length, embedding_dim)
            
        Raises:
            ValueError: If sequence is invalid
        """
        # Validate sequence
        validate_sequence(sequence)
        
        # Tokenize the sequence
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024  # ESM2 max sequence length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            # Extract token embeddings (exclude special tokens)
            embeddings = outputs.last_hidden_state[0, 1:-1, :]  # Remove [CLS] and [SEP]
        
        return embeddings
    
    def average_binding_site_embeddings(self, 
                                      embeddings: torch.Tensor, 
                                      binding_site_indices: List[int]) -> torch.Tensor:
        """
        Average the embeddings of binding site residues.
        
        Args:
            embeddings: Full sequence embeddings (sequence_length, embedding_dim)
            binding_site_indices: List of residue indices (1-based)
            
        Returns:
            Averaged embedding tensor of shape (embedding_dim,)
            
        Raises:
            ValueError: If no valid binding site indices found
        """
        # Validate indices
        validate_indices(binding_site_indices, embeddings.shape[0])
        
        # Convert to 0-based indexing and validate
        zero_based_indices = [idx - 1 for idx in binding_site_indices]
        valid_indices = [idx for idx in zero_based_indices 
                        if 0 <= idx < embeddings.shape[0]]
        
        if not valid_indices:
            raise ValueError("No valid binding site indices found")
        
        if len(valid_indices) != len(binding_site_indices):
            logger.warning(f"{len(binding_site_indices) - len(valid_indices)} indices were out of range")
        
        # Extract binding site embeddings and average
        binding_site_embeddings = embeddings[valid_indices]
        averaged_embedding = torch.mean(binding_site_embeddings, dim=0)
        
        return averaged_embedding
    
    def predict(self, 
                sequence: str, 
                binding_site_indices: List[int]) -> Dict[str, float]:
        """
        Predict binding site class probabilities.
        
        Args:
            sequence: Protein sequence string
            binding_site_indices: List of binding site residue indices (1-based)
            
        Returns:
            Dictionary with class names and their probabilities
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        if not binding_site_indices:
            raise ValueError("Binding site indices cannot be empty")
        
        # Get ESM2 embeddings
        embeddings = self.get_esm2_embeddings(sequence)
        
        # Average binding site embeddings
        binding_site_embedding = self.average_binding_site_embeddings(
            embeddings, binding_site_indices
        )
        
        # Predict with the model
        self.prediction_model.eval()
        with torch.no_grad():
            logits = self.prediction_model(binding_site_embedding.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
        
        # Create results dictionary
        results = {}
        for i, class_name in enumerate(self.classes):
            results[class_name] = float(probabilities[i])
        
        return results
    
    def predict_batch(self, 
                     sequences: List[str], 
                     binding_sites_list: List[List[int]],
                     batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Predict binding site classes for multiple sequences.
        
        Args:
            sequences: List of protein sequences
            binding_sites_list: List of binding site indices for each sequence
            batch_size: Number of sequences to process at once
            
        Returns:
            List of prediction dictionaries
        """
        if len(sequences) != len(binding_sites_list):
            raise ValueError("Number of sequences and binding sites must match")
        
        results = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_binding_sites = binding_sites_list[i:i + batch_size]
            
            batch_results = []
            for seq, sites in zip(batch_sequences, batch_binding_sites):
                try:
                    result = self.predict(seq, sites)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing sequence {i}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results
    
    def save_model(self, path: Union[str, Path]):
        """
        Save the trained prediction model.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state and metadata
        checkpoint = {
            'model_state_dict': self.prediction_model.state_dict(),
            'classes': self.classes,
            'embedding_dim': self.embedding_dim,
            'model_architecture': {
                'input_dim': self.embedding_dim,
                'num_classes': self.num_classes
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """
        Load a trained prediction model.
        
        Args:
            path: Path to the model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model architecture doesn't match
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Validate compatibility
            if checkpoint.get('embedding_dim') != self.embedding_dim:
                logger.warning("Embedding dimensions don't match. This might cause issues.")
            
            if checkpoint.get('classes') != self.classes:
                logger.warning("Class labels don't match. This might cause issues.")
            
            # Load model state
            self.prediction_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def print_prediction(self, 
                        sequence: str, 
                        binding_site_indices: List[int],
                        show_details: bool = True):
        """
        Print prediction results in a formatted way.
        
        Args:
            sequence: Protein sequence string
            binding_site_indices: List of binding site residue indices (1-based)
            show_details: Whether to show detailed information
        """
        try:
            if show_details:
                print(f"\nProtein Sequence: {sequence}")
                print(f"Binding Site Indices: {binding_site_indices}")
                print(f"Sequence Length: {len(sequence)}")
                print("-" * 50)
            
            results = self.predict(sequence, binding_site_indices)
            
            print("Binding Site Classification Probabilities:")
            print("-" * 40)
            
            # Sort by probability (descending)
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            
            for class_name, probability in sorted_results:
                print(f"{class_name:>10}: {probability:.4f} ({probability*100:.2f}%)")
            
            print("-" * 40)
            predicted_class = sorted_results[0][0]
            confidence = sorted_results[0][1]
            print(f"Predicted Class: {predicted_class} (confidence: {confidence:.4f})")
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            print(f"Error: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models.
        
        Returns:
            Dictionary with model information
        """
        return {
            'esm_model': self.esm_model.config.name_or_path,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'device': str(self.device),
            'prediction_model_params': sum(p.numel() for p in self.prediction_model.parameters())
        }