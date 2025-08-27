"""
Test suite for the Protein Binding Site Classifier.
"""

import unittest
import tempfile
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from protein_classifier import (
    ProteinBindingSiteClassifier,
    PredictionModel,
    validate_sequence,
    validate_indices,
    clean_sequence,
    create_example_data
)


class TestProteinBindingSiteClassifier(unittest.TestCase):
    """Test cases for the main classifier."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.classifier = ProteinBindingSiteClassifier()
        cls.test_sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQ"
        cls.test_binding_sites = [10, 20, 30, 40]
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsInstance(self.classifier, ProteinBindingSiteClassifier)
        self.assertEqual(len(self.classifier.classes), 5)
        self.assertIn('phosphate', self.classifier.classes)
        self.assertIsNotNone(self.classifier.esm_model)
        self.assertIsNotNone(self.classifier.prediction_model)
    
    def test_get_esm2_embeddings(self):
        """Test ESM2 embedding generation."""
        embeddings = self.classifier.get_esm2_embeddings(self.test_sequence)
        
        # Check shape
        self.assertEqual(embeddings.shape[0], len(self.test_sequence))
        self.assertEqual(embeddings.shape[1], self.classifier.embedding_dim)
        
        # Check data type
        self.assertEqual(embeddings.dtype, torch.float32)
    
    def test_average_binding_site_embeddings(self):
        """Test binding site embedding averaging."""
        embeddings = self.classifier.get_esm2_embeddings(self.test_sequence)
        averaged = self.classifier.average_binding_site_embeddings(
            embeddings, self.test_binding_sites
        )
        
        # Check shape
        self.assertEqual(averaged.shape[0], self.classifier.embedding_dim)
        self.assertEqual(len(averaged.shape), 1)
    
    def test_predict(self):
        """Test prediction functionality."""
        results = self.classifier.predict(self.test_sequence, self.test_binding_sites)
        
        # Check result format
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 5)
        
        # Check all classes present
        for class_name in self.classifier.classes:
            self.assertIn(class_name, results)
        
        # Check probabilities sum to 1
        total_prob = sum(results.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
        # Check all probabilities are between 0 and 1
        for prob in results.values():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        sequences = [self.test_sequence, self.test_sequence[:50]]
        binding_sites_list = [[10, 20, 30], [5, 15, 25]]
        
        results = self.classifier.predict_batch(sequences, binding_sites_list)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], dict)
        self.assertIsInstance(results[1], dict)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.pth"
            
            # Save model
            self.classifier.save_model(model_path)
            self.assertTrue(model_path.exists())
            
            # Create new classifier and load model
            new_classifier = ProteinBindingSiteClassifier()
            new_classifier.load_model(model_path)
            
            # Test that loaded model gives same results
            results1 = self.classifier.predict(self.test_sequence, self.test_binding_sites)
            results2 = new_classifier.predict(self.test_sequence, self.test_binding_sites)
            
            for class_name in self.classifier.classes:
                self.assertAlmostEqual(results1[class_name], results2[class_name], places=5)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.classifier.get_model_info()
        
        required_keys = ['esm_model', 'embedding_dim', 'num_classes', 'classes', 'device']
        for key in required_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['num_classes'], 5)
        self.assertEqual(len(info['classes']), 5)


class TestPredictionModel(unittest.TestCase):
    """Test cases for the prediction model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 320
        self.num_classes = 5
        self.model = PredictionModel(self.input_dim, self.num_classes)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, PredictionModel)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.num_classes, self.num_classes)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, self.input_dim)
        
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        self.assertEqual(output.dtype, torch.float32)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        batch_size = 4
        x = torch.randn(batch_size, self.input_dim)
        
        probs = self.model.predict_proba(x)
        
        self.assertEqual(probs.shape, (batch_size, self.num_classes))
        
        # Check probabilities sum to 1
        prob_sums = torch.sum(probs, dim=1)
        for prob_sum in prob_sums:
            self.assertAlmostEqual(prob_sum.item(), 1.0, places=5)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        required_keys = ['input_dim', 'num_classes', 'total_parameters', 'trainable_parameters']
        for key in required_keys:
            self.assertIn(key, info)
        
        self.assertGreater(info['total_parameters'], 0)
        self.assertEqual(info['total_parameters'], info['trainable_parameters'])


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_validate_sequence(self):
        """Test sequence validation."""
        # Valid sequences
        validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        validate_sequence("MKVLWAALL")
        
        # Invalid sequences
        with self.assertRaises(ValueError):
            validate_sequence("")  # Empty sequence
        
        with self.assertRaises(ValueError):
            validate_sequence("MKVLWAALLZ", allow_extended=False)  # Invalid AA
        
        with self.assertRaises(ValueError):
            validate_sequence(123)  # Not a string
    
    def test_validate_indices(self):
        """Test binding site indices validation."""
        # Valid indices
        validate_indices([1, 2, 3, 4], 10)
        validate_indices([5, 10], 15)
        
        # Invalid indices
        with self.assertRaises(ValueError):
            validate_indices([], 10)  # Empty list
        
        with self.assertRaises(ValueError):
            validate_indices([0, 1, 2], 10)  # Zero index
        
        with self.assertRaises(ValueError):
            validate_indices([1, 2, 15], 10)  # Out of range
        
        with self.assertRaises(ValueError):
            validate_indices([1.5, 2], 10)  # Non-integer
    
    def test_clean_sequence(self):
        """Test sequence cleaning."""
        # Test whitespace removal
        self.assertEqual(clean_sequence("M K V L"), "MKVL")
        
        # Test case conversion
        self.assertEqual(clean_sequence("mkvl"), "MKVL")
        
        # Test mixed case and whitespace
        self.assertEqual(clean_sequence("m k V l \n\t"), "MKVL")
        
        # Test with invalid characters
        self.assertEqual(clean_sequence("MKV123L"), "MKVL")
    
    def test_create_example_data(self):
        """Test example data creation."""
        example_data = create_example_data()
        
        self.assertIsInstance(example_data, dict)
        self.assertIn('sequences', example_data)
        self.assertIn('metadata', example_data)
        
        sequences = example_data['sequences']
        self.assertIsInstance(sequences, list)
        self.assertGreater(len(sequences), 0)
        
        # Check first sequence structure
        seq = sequences[0]
        required_keys = ['id', 'sequence', 'binding_sites', 'label']
        for key in required_keys:
            self.assertIn(key, seq)
        
        # Validate sequence and binding sites
        validate_sequence(seq['sequence'])
        validate_indices(seq['binding_sites'], len(seq['sequence']))


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = ProteinBindingSiteClassifier()
    
    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        with self.assertRaises(ValueError):
            self.classifier.predict("", [1, 2, 3])
    
    def test_empty_binding_sites(self):
        """Test handling of empty binding sites."""
        with self.assertRaises(ValueError):
            self.classifier.predict("MKVLWAALL", [])
    
    def test_invalid_binding_sites(self):
        """Test handling of invalid binding sites."""
        sequence = "MKVLWAALL"
        
        # Out of range indices
        with self.assertRaises(ValueError):
            self.classifier.predict(sequence, [1, 2, 20])
        
        # Zero or negative indices
        with self.assertRaises(ValueError):
            self.classifier.predict(sequence, [0, 1, 2])
    
    def test_very_long_sequence(self):
        """Test handling of very long sequences."""
        long_sequence = "A" * 3000  # Very long sequence
        binding_sites = [100, 200, 300]
        
        # Should work but might be slow
        try:
            results = self.classifier.predict(long_sequence, binding_sites)
            self.assertIsInstance(results, dict)
        except Exception as e:
            # If it fails due to memory, that's acceptable
            self.assertIn("memory", str(e).lower())
    
    def test_single_residue_binding_site(self):
        """Test handling of single residue binding site."""
        sequence = "MKVLWAALL"
        binding_sites = [5]  # Single residue
        
        results = self.classifier.predict(sequence, binding_sites)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize classifier
        classifier = ProteinBindingSiteClassifier()
        
        # Get example data
        example_data = create_example_data()
        
        # Process each example
        for seq_data in example_data['sequences']:
            # Make prediction
            results = classifier.predict(
                seq_data['sequence'], 
                seq_data['binding_sites']
            )
            
            # Verify results
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 5)
            
            # Check that probabilities are valid
            for prob in results.values():
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)
            
            # Check that probabilities sum to 1
            total_prob = sum(results.values())
            self.assertAlmostEqual(total_prob, 1.0, places=5)
    
    def test_batch_processing_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        classifier = ProteinBindingSiteClassifier()
        
        sequences = ["MKVLWAALL", "ACDEFGHIK", "LMNPQRSTVWY"]
        binding_sites_list = [[1, 5, 9], [2, 4, 6], [3, 7, 11]]
        
        # Individual predictions
        individual_results = []
        for seq, sites in zip(sequences, binding_sites_list):
            result = classifier.predict(seq, sites)
            individual_results.append(result)
        
        # Batch prediction
        batch_results = classifier.predict_batch(sequences, binding_sites_list)
        
        # Compare results
        self.assertEqual(len(individual_results), len(batch_results))
        
        for ind_result, batch_result in zip(individual_results, batch_results):
            for class_name in classifier.classes:
                self.assertAlmostEqual(
                    ind_result[class_name], 
                    batch_result[class_name], 
                    places=5
                )


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)