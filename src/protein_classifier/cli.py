"""
Command line interface for the Protein Binding Site Classifier.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .classifier import ProteinBindingSiteClassifier
from .utils import (
    load_sequences_from_file,
    save_predictions_to_file,
    create_example_data,
    format_prediction_output
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_binding_sites(binding_sites_str: str) -> List[int]:
    """
    Parse binding site indices from command line string.
    
    Args:
        binding_sites_str: Comma-separated string of indices
        
    Returns:
        List of integer indices
    """
    try:
        return [int(idx.strip()) for idx in binding_sites_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid binding site indices: {e}")


def predict_single(args) -> None:
    """Handle single sequence prediction."""
    logger.info("Processing single sequence...")
    
    # Parse binding sites
    binding_sites = parse_binding_sites(args.binding_sites)
    
    # Initialize classifier
    classifier = ProteinBindingSiteClassifier(
        model_path=args.model_path,
        device=args.device
    )
    
    # Make prediction
    try:
        results = classifier.predict(args.sequence, binding_sites)
        
        if args.output_file:
            # Save to file
            output_data = {
                "sequence": args.sequence,
                "binding_sites": binding_sites,
                "predictions": results
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Results saved to {args.output_file}")
        
        if not args.quiet:
            print(format_prediction_output(results))
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


def predict_batch(args) -> None:
    """Handle batch prediction from file."""
    logger.info(f"Processing batch file: {args.input_file}")
    
    try:
        # Load input data
        data = load_sequences_from_file(args.input_file)
        sequences_data = data['sequences']
        
        # Initialize classifier
        classifier = ProteinBindingSiteClassifier(
            model_path=args.model_path,
            device=args.device
        )
        
        # Process sequences
        results = []
        total = len(sequences_data)
        
        for i, seq_data in enumerate(sequences_data):
            if not args.quiet:
                print(f"Processing sequence {i+1}/{total}...", end='\r')
            
            try:
                prediction = classifier.predict(
                    seq_data['sequence'], 
                    seq_data['binding_sites']
                )
                
                result = {
                    'id': seq_data.get('id', f'seq_{i+1}'),
                    'sequence': seq_data['sequence'],
                    'binding_sites': seq_data['binding_sites'],
                    'predictions': prediction,
                    'true_label': seq_data.get('label'),
                    'predicted_label': max(prediction, key=prediction.get)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing sequence {i+1}: {e}")
                results.append({
                    'id': seq_data.get('id', f'seq_{i+1}'),
                    'error': str(e)
                })
        
        # Save results
        if args.output_file:
            save_predictions_to_file(results, args.output_file)
        else:
            print(json.dumps(results, indent=2))
        
        logger.info(f"Processed {len(results)} sequences")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


def create_example(args) -> None:
    """Create example data file."""
    logger.info("Creating example data...")
    
    example_data = create_example_data()
    
    output_file = args.output_file or "example_sequences.json"
    
    with open(output_file, 'w') as f:
        json.dump(example_data, f, indent=2)
    
    logger.info(f"Example data saved to {output_file}")
    print(f"Example file created: {output_file}")
    print("You can use this file with: python -m protein_classifier batch --input_file example_sequences.json")


def show_info(args) -> None:
    """Show model and system information."""
    try:
        classifier = ProteinBindingSiteClassifier(
            model_path=args.model_path,
            device=args.device
        )
        
        info = classifier.get_model_info()
        
        print("=== Protein Binding Site Classifier Info ===")
        print(f"ESM Model: {info['esm_model']}")
        print(f"Embedding Dimension: {info['embedding_dim']}")
        print(f"Number of Classes: {info['num_classes']}")
        print(f"Classes: {', '.join(info['classes'])}")
        print(f"Device: {info['device']}")
        print(f"Prediction Model Parameters: {info['prediction_model_params']:,}")
        
        # Memory estimation for different sequence lengths
        from .utils import estimate_memory_usage
        
        print("\n=== Memory Usage Estimates ===")
        for seq_len in [100, 500, 1000, 2000]:
            mem_est = estimate_memory_usage(seq_len)
            print(f"Sequence length {seq_len}: ~{mem_est['total_estimated_mb']:.1f} MB")
        
    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Protein Binding Site Classifier - Predict binding site types using ESM2 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python -m protein_classifier single --sequence "MKVLWAALL..." --binding_sites "1,2,3,4,5"
  
  # Batch prediction
  python -m protein_classifier batch --input_file sequences.json --output_file results.json
  
  # Create example data
  python -m protein_classifier example --output_file my_examples.json
  
  # Show system info
  python -m protein_classifier info
        """
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--device', help='Computing device (cuda/cpu)')
    parser.add_argument('--model_path', help='Path to trained prediction model')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single prediction command
    single_parser = subparsers.add_parser('single', help='Predict single sequence')
    single_parser.add_argument('--sequence', required=True, help='Protein sequence')
    single_parser.add_argument('--binding_sites', required=True, 
                              help='Comma-separated binding site indices (1-based)')
    single_parser.add_argument('--output_file', help='Output JSON file')
    
    # Batch prediction command
    batch_parser = subparsers.add_parser('batch', help='Predict multiple sequences from file')
    batch_parser.add_argument('--input_file', required=True, help='Input JSON file')
    batch_parser.add_argument('--output_file', help='Output JSON file')
    
    # Example creation command
    example_parser = subparsers.add_parser('example', help='Create example data file')
    example_parser.add_argument('--output_file', help='Output file name')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model and system information')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'single':
            predict_single(args)
        elif args.command == 'batch':
            predict_batch(args)
        elif args.command == 'example':
            create_example(args)
        elif args.command == 'info':
            show_info(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()