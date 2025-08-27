"""
Batch processing example for the Protein Binding Site Classifier.

This script demonstrates how to process multiple protein sequences
efficiently using batch processing capabilities.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add src to path for running example
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from protein_classifier import (
    ProteinBindingSiteClassifier,
    load_sequences_from_file,
    save_predictions_to_file,
    batch_process_sequences,
    calculate_metrics,
    create_example_data
)


def batch_prediction_example():
    """Demonstrate batch prediction with multiple sequences."""
    print("=== Batch Prediction Example ===\n")
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = ProteinBindingSiteClassifier()
    
    # Create or load example data
    example_file = "temp_example_data.json"
    
    # Create example data if it doesn't exist
    if not os.path.exists(example_file):
        print("Creating example data...")
        example_data = create_example_data()
        with open(example_file, 'w') as f:
            json.dump(example_data, f, indent=2)
    
    # Load data
    print(f"Loading sequences from {example_file}...")
    data = load_sequences_from_file(example_file)
    sequences_data = data['sequences']
    
    print(f"Loaded {len(sequences_data)} sequences\n")
    
    # Extract sequences and binding sites
    sequences = [seq_data['sequence'] for seq_data in sequences_data]
    binding_sites_list = [seq_data['binding_sites'] for seq_data in sequences_data]
    true_labels = [seq_data['label'] for seq_data in sequences_data]
    
    # Method 1: Using classifier's built-in batch processing
    print("Method 1: Using classifier.predict_batch()")
    start_time = time.time()
    
    batch_results = classifier.predict_batch(
        sequences, 
        binding_sites_list, 
        batch_size=4
    )
    
    batch_time = time.time() - start_time
    print(f"Batch processing completed in {batch_time:.2f} seconds\n")
    
    # Method 2: Using utility function with progress bar
    print("Method 2: Using batch_process_sequences() with progress")
    start_time = time.time()
    
    util_results = batch_process_sequences(
        sequences,
        binding_sites_list,
        classifier,
        batch_size=2,
        show_progress=True
    )
    
    util_time = time.time() - start_time
    print(f"Utility batch processing completed in {util_time:.2f} seconds\n")
    
    # Process results
    predictions = []
    predicted_labels = []
    
    for i, (seq_data, result) in enumerate(zip(sequences_data, batch_results)):
        if result is not None:
            predicted_label = max(result, key=result.get)
            predicted_labels.append(predicted_label)
            
            prediction_entry = {
                'id': seq_data['id'],
                'sequence_length': len(seq_data['sequence']),
                'num_binding_sites': len(seq_data['binding_sites']),
                'true_label': seq_data['label'],
                'predicted_label': predicted_label,
                'confidence': result[predicted_label],
                'all_probabilities': result,
                'description': seq_data.get('description', '')
            }
            predictions.append(prediction_entry)
        else:
            print(f"Error processing sequence {i+1}")
            predicted_labels.append('error')
    
    # Display results
    print("=== Prediction Results ===")
    print(f"{'ID':<25} {'True':<10} {'Predicted':<10} {'Confidence':<10} {'Status'}")
    print("-" * 70)
    
    for pred in predictions:
        status = "✓" if pred['true_label'] == pred['predicted_label'] else "✗"
        print(f"{pred['id']:<25} {pred['true_label']:<10} {pred['predicted_label']:<10} "
              f"{pred['confidence']:<10.3f} {status}")
    
    print()
    
    # Calculate metrics
    print("=== Performance Metrics ===")
    try:
        # Filter out error cases for metrics calculation
        valid_true = [true for true, pred in zip(true_labels, predicted_labels) if pred != 'error']
        valid_pred = [pred for pred in predicted_labels if pred != 'error']
        
        if valid_true and valid_pred:
            metrics = calculate_metrics(valid_true, valid_pred, classifier.classes)
            
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Macro F1: {metrics['macro_f1']:.3f}")
            print(f"Macro Precision: {metrics['macro_precision']:.3f}")
            print(f"Macro Recall: {metrics['macro_recall']:.3f}")
            
            print("\nPer-class metrics:")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                print(f"  {class_name}:")
                print(f"    Precision: {class_metrics['precision']:.3f}")
                print(f"    Recall: {class_metrics['recall']:.3f}")
                print(f"    F1: {class_metrics['f1']:.3f}")
                print(f"    Support: {class_metrics['support']}")
        else:
            print("No valid predictions for metrics calculation")
    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    print()
    
    # Save results
    output_file = "batch_prediction_results.json"
    print(f"Saving results to {output_file}...")
    
    output_data = {
        'predictions': predictions,
        'metadata': {
            'processing_time': batch_time,
            'num_sequences': len(sequences_data),
            'batch_size': 4,
            'model_info': classifier.get_model_info()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Clean up temporary file
    if os.path.exists(example_file):
        os.remove(example_file)
    
    print(f"Results saved to {output_file}")
    print("Batch processing example completed!")


def performance_comparison():
    """Compare individual vs batch processing performance."""
    print("\n=== Performance Comparison ===\n")
    
    classifier = ProteinBindingSiteClassifier()
    
    # Generate test data
    sequences = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQ" * 2,  # ~80 residues
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY" * 3,  # ~120 residues
        "MTMDKSELVQKAKLAEQAERYDEMVESMKKVAGMDVELT" * 4,   # ~160 residues
        "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMI" * 5,   # ~200 residues
    ]
    
    binding_sites_list = [
        [5, 15, 25, 35, 45],
        [10, 30, 50, 70, 90],
        [20, 40, 80, 120, 140],
        [25, 75, 125, 175, 195]
    ]
    
    print(f"Testing with {len(sequences)} sequences")
    print("Sequence lengths:", [len(seq) for seq in sequences])
    print()
    
    # Individual processing
    print("Individual processing...")
    start_time = time.time()
    
    individual_results = []
    for seq, sites in zip(sequences, binding_sites_list):
        result = classifier.predict(seq, sites)
        individual_results.append(result)
    
    individual_time = time.time() - start_time
    
    # Batch processing
    print("Batch processing...")
    start_time = time.time()
    
    batch_results = classifier.predict_batch(sequences, binding_sites_list)
    
    batch_time = time.time() - start_time
    
    # Results
    print(f"\nTiming Results:")
    print(f"Individual processing: {individual_time:.3f} seconds")
    print(f"Batch processing: {batch_time:.3f} seconds")
    print(f"Speedup: {individual_time/batch_time:.2f}x")
    
    # Verify results are the same
    results_match = True
    for i, (ind_result, batch_result) in enumerate(zip(individual_results, batch_results)):
        if batch_result is None:
            results_match = False
            print(f"Batch result {i} is None")
            continue
            
        for class_name in classifier.classes:
            if abs(ind_result[class_name] - batch_result[class_name]) > 1e-6:
                results_match = False
                print(f"Results differ for sequence {i}, class {class_name}")
    
    print(f"Results consistency: {'✓ Match' if results_match else '✗ Differ'}")


def memory_usage_example():
    """Demonstrate memory usage estimation and optimization."""
    print("\n=== Memory Usage Example ===\n")
    
    from protein_classifier.utils import estimate_memory_usage
    
    # Test different sequence lengths
    test_lengths = [100, 500, 1000, 2000, 5000]
    batch_sizes = [1, 4, 8]
    
    print("Memory usage estimates (MB):")
    print(f"{'Seq Length':<12} {'Batch=1':<10} {'Batch=4':<10} {'Batch=8':<10}")
    print("-" * 50)
    
    for length in test_lengths:
        estimates = []
        for batch_size in batch_sizes:
            mem_est = estimate_memory_usage(length, batch_size)
            estimates.append(mem_est['total_estimated_mb'])
        
        print(f"{length:<12} {estimates[0]:<10.1f} {estimates[1]:<10.1f} {estimates[2]:<10.1f}")
    
    print("\nRecommendations:")
    print("- Use smaller batch sizes for very long sequences (>2000 residues)")
    print("- Monitor GPU memory usage when processing multiple sequences")
    print("- Consider CPU processing for very large batches if GPU memory is limited")


if __name__ == "__main__":
    print("Protein Binding Site Classifier - Batch Processing Examples")
    print("=" * 65)
    
    try:
        # Run examples
        batch_prediction_example()
        performance_comparison()
        memory_usage_example()
        
        print("\nAll batch processing examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
