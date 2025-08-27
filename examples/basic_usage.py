"""
Basic usage examples for the Protein Binding Site Classifier.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from protein_classifier import ProteinBindingSiteClassifier, create_example_data


def example_1_basic_prediction():
    """Example 1: Basic single prediction."""
    print("=== Example 1: Basic Prediction ===")
    
    # Initialize classifier
    classifier = ProteinBindingSiteClassifier()
    
    # Example protein sequence (ATP synthase subunit)
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ"
    binding_sites = [23, 45, 67, 89]  # Example binding site residues
    
    print(f"Sequence: {sequence}")
    print(f"Binding sites: {binding_sites}")
    print()
    
    # Make prediction
    results = classifier.predict(sequence, binding_sites)
    
    # Display results
    print("Prediction probabilities:")
    for class_name, probability in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {probability:.4f} ({probability*100:.2f}%)")
    
    predicted_class = max(results, key=results.get)
    print(f"\nPredicted class: {predicted_class}")
    print()


def example_2_batch_prediction():
    """Example 2: Batch prediction with multiple sequences."""
    print("=== Example 2: Batch Prediction ===")
    
    classifier = ProteinBindingSiteClassifier()
    
    # Multiple sequences and their binding sites
    sequences = [
        "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ",
        "MTMDKSELVQKAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIRE",
        "MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNG"
    ]
    
    binding_sites_list = [
        [23, 45, 67, 89],
        [12, 34, 56, 78],
        [15, 35, 55, 75, 95]
    ]
    
    # Batch prediction
    batch_results = classifier.predict_batch(sequences, binding_sites_list)
    
    # Display results
    for i, results in enumerate(batch_results):
        if results is not None:
            predicted_class = max(results, key=results.get)
            confidence = results[predicted_class]
            print(f"Sequence {i+1}: {predicted_class} (confidence: {confidence:.3f})")
        else:
            print(f"Sequence {i+1}: Error in prediction")
    print()


def example_3_using_example_data():
    """Example 3: Using built-in example data."""
    print("=== Example 3: Using Example Data ===")
    
    # Create example data
    example_data = create_example_data()
    
    classifier = ProteinBindingSiteClassifier()
    
    # Process example sequences
    for seq_data in example_data['sequences']:
        print(f"Processing {seq_data['id']} ({seq_data['description']})")
        
        results = classifier.predict(seq_data['sequence'], seq_data['binding_sites'])
        predicted_class = max(results, key=results.get)
        true_label = seq_data['label']
        
        print(f"  True label: {true_label}")
        print(f"  Predicted: {predicted_class}")
        print(f"  Confidence: {results[predicted_class]:.3f}")
        print(f"  Correct: {'✓' if predicted_class == true_label else '✗'}")
        print()


def example_4_model_information():
    """Example 4: Getting model information."""
    print("=== Example 4: Model Information ===")
    
    classifier = ProteinBindingSiteClassifier()
    
    # Get model info
    info = classifier.get_model_info()
    
    print("Model Information:")
    for key, value in info.items():
        if key == 'classes':
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")
    print()


def example_5_error_handling():
    """Example 5: Error handling and validation."""
    print("=== Example 5: Error Handling ===")
    
    classifier = ProteinBindingSiteClassifier()
    
    # Test various error conditions
    test_cases = [
        ("", [1, 2, 3], "Empty sequence"),
        ("MKVLWAALL", [], "Empty binding sites"),
        ("MKVLWAALL", [50, 60], "Indices out of range"),
        ("MKVLWAALLX", [1, 2], "Invalid amino acid"),
    ]
    
    for sequence, binding_sites, description in test_cases:
        print(f"Testing: {description}")
        try:
            results = classifier.predict(sequence, binding_sites)
            print(f"  Success: {max(results, key=results.get)}")
        except Exception as e:
            print(f"  Error: {e}")
        print()


def example_6_custom_display():
    """Example 6: Custom result display."""
    print("=== Example 6: Custom Display ===")
    
    classifier = ProteinBindingSiteClassifier()
    
    sequence = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELL"
    binding_sites = [10, 20, 30, 40, 50]
    
    results = classifier.predict(sequence, binding_sites)
    
    # Custom formatted output
    print(f"Sequence length: {len(sequence)}")
    print(f"Binding site positions: {binding_sites}")
    print(f"Number of binding site residues: {len(binding_sites)}")
    print()
    
    print("Detailed Results:")
    print("-" * 50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (class_name, probability) in enumerate(sorted_results):
        status = "★" if i == 0 else " "
        bar_length = int(probability * 30)  # Scale to 30 characters
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        print(f"{status} {class_name:>10} │{bar}│ {probability:.3f}")
    
    print("-" * 50)
    print()


if __name__ == "__main__":
    print("Protein Binding Site Classifier - Basic Usage Examples")
    print("=" * 60)
    print()
    
    # Run all examples
    example_1_basic_prediction()
    example_2_batch_prediction()
    example_3_using_example_data()
    example_4_model_information()
    example_5_error_handling()
    example_6_custom_display()
    
    print("All examples completed successfully!")

    
