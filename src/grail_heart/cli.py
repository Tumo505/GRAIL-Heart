"""
GRAIL-Heart Command Line Interface

Usage:
    grail-heart predict <input_file> [options]
    grail-heart info
    grail-heart app

Examples:
    # Run forward prediction
    grail-heart predict my_data.h5ad --output results.csv
    
    # Run inverse modeling
    grail-heart predict my_data.h5ad --mode inverse --output causal_results.csv
    
    # Start web app
    grail-heart app
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="grail-heart",
        description="GRAIL-Heart: Cardiac L-R Interaction Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  grail-heart predict my_data.h5ad --output results.csv
  grail-heart predict my_data.h5ad --mode inverse
  grail-heart app
  grail-heart info
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run L-R interaction prediction on scRNA-seq data"
    )
    predict_parser.add_argument(
        "input",
        type=str,
        help="Input file (h5ad, h5, or csv format)"
    )
    predict_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (CSV format)"
    )
    predict_parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["forward", "inverse"],
        default="forward",
        help="Prediction mode: forward (expression→L-R) or inverse (fate→causal L-R)"
    )
    predict_parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=100,
        help="Number of top L-R pairs to return (default: 100)"
    )
    predict_parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)"
    )
    predict_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda/cpu)"
    )
    predict_parser.add_argument(
        "--network-output",
        type=str,
        default=None,
        help="Output file for network JSON"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show model information"
    )
    
    # App command
    app_parser = subparsers.add_parser(
        "app",
        help="Start the web application"
    )
    app_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for web server (default: 8501)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "predict":
        run_predict(args)
    elif args.command == "info":
        run_info(args)
    elif args.command == "app":
        run_app(args)


def run_predict(args):
    """Run prediction on input data."""
    from grail_heart import load_pretrained
    
    print("=" * 60)
    print("GRAIL-Heart Prediction")
    print("=" * 60)
    
    # Load model
    try:
        predictor = load_pretrained(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide --checkpoint path or download the pretrained model.")
        sys.exit(1)
    
    # Run prediction
    results = predictor.predict(
        data=args.input,
        mode=args.mode,
        top_n=args.top_n,
    )
    
    # Save results
    if args.output:
        results.to_csv(args.output)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nTop 10 L-R Interactions:")
        print(results.top_lr_pairs.head(10).to_string())
    
    if args.network_output:
        results.to_json(args.network_output)
        print(f"Network saved to: {args.network_output}")


def run_info(args):
    """Show model information."""
    print("=" * 60)
    print("GRAIL-Heart Model Information")
    print("=" * 60)
    print()
    print("Version: 1.0.0")
    print("Model: Graph Neural Network for Cardiac L-R Analysis")
    print()
    print("Features:")
    print("  - Forward Modeling: Expression → L-R predictions")
    print("  - Inverse Modeling: Observed fates → Causal L-R signals")
    print("  - Mechanosensitive pathway analysis")
    print()
    print("Supported Input Formats:")
    print("  - h5ad: AnnData format (recommended)")
    print("  - h5: 10X Genomics format")
    print("  - csv: Gene expression matrix")
    print()
    print("Links:")
    print("  GitHub: https://github.com/tumo505/GRAIL-Heart")
    print("  Explorer: https://tumo505.github.io/GRAIL-Heart/outputs/cytoscape/index.html")
    print()
    
    # Try to load model for more info
    try:
        from grail_heart import load_pretrained
        predictor = load_pretrained()
        n_params = sum(p.numel() for p in predictor.model.parameters())
        print(f"Checkpoint: Loaded ({n_params/1e9:.2f}B parameters)")
        print(f"Inverse Modeling: {'Enabled' if predictor.has_inverse else 'Disabled'}")
    except:
        print("Checkpoint: Not found (run with --checkpoint to specify)")


def run_app(args):
    """Start the web application."""
    import subprocess
    
    app_path = Path(__file__).parent.parent.parent / "app" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Starting GRAIL-Heart Web Application")
    print("=" * 60)
    print(f"URL: http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print()
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
    ])


if __name__ == "__main__":
    main()
