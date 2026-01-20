"""
Phase 7 Evaluation CLI

Command-line interface for running Phase 7 analysis and generating reports.
"""

import argparse
from pathlib import Path
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 7 Evaluation: NS-MAS Analysis and Report Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default settings
  python -m src.evaluation.cli --results-dir results --output-dir results/analysis

  # Generate only visualizations
  python -m src.evaluation.cli --results-dir results --plots-only

  # Run with fewer bootstrap samples for faster testing
  python -m src.evaluation.cli --results-dir results --bootstrap 1000

  # Show plots interactively
  python -m src.evaluation.cli --results-dir results --show-plots
""",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing experiment results (default: results/)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory for analysis outputs (default: results/analysis/)",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for CIs (default: 10000)",
    )

    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate visualizations, skip statistical analysis",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate markdown report, skip plots",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (requires display)",
    )

    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also generate LaTeX tables for paper",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Import modules (deferred to avoid slow imports for --help)
    from .data_loader import DataLoader
    from .report import ReportGenerator
    from .visualization import Visualizer

    # Validate results directory
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 7 Evaluation: NS-MAS Analysis")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bootstrap samples: {args.bootstrap}")
    print()

    # Load data
    print("Loading experiment data...")
    loader = DataLoader(args.results_dir)
    loader.load_all()

    if args.verbose:
        print(loader.summary())
        print()

    if not loader.systems:
        print("Error: No experiment data found!")
        sys.exit(1)

    print(f"Loaded {len(loader.systems)} systems")

    # Initialize report generator
    report_gen = ReportGenerator(loader, output_dir=args.output_dir)

    # Run analysis
    if not args.plots_only:
        print()
        print("Running statistical analysis...")
        results = report_gen.run_full_analysis(n_bootstrap=args.bootstrap)

        print("Generating markdown report...")
        report = report_gen.generate_markdown_report(results)
        report_path = args.output_dir / "phase7_analysis_report.md"
        print(f"  Report saved to: {report_path}")

        if args.latex:
            print("Generating LaTeX tables...")
            tables = report_gen.generate_latex_tables(results)
            print(f"  LaTeX tables saved to: {args.output_dir / 'tables'}")

    # Generate visualizations
    if not args.report_only:
        print()
        print("Generating visualizations...")

        try:
            viz = Visualizer(output_dir=args.output_dir / "plots")

            # Need to run analysis if we only want plots
            if args.plots_only:
                results = report_gen.run_full_analysis(n_bootstrap=args.bootstrap)

            paths = viz.generate_all_plots(
                robustness=results.robustness,
                complexity=results.complexity,
                cost=results.cost,
                errors=results.errors,
                self_correction=results.self_correction,
                show=args.show_plots,
            )

            for name, path in paths.items():
                print(f"  {name}: {path}")

        except ImportError as e:
            print(f"Warning: Could not generate plots: {e}")
            print("Install matplotlib and seaborn for visualization support.")

    print()
    print("=" * 60)
    print("Phase 7 Analysis Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review the analysis report in results/analysis/")
    print("  2. Check visualizations in results/analysis/plots/")
    print("  3. Use LaTeX tables in your EXTRAAMAS paper")
    print("  4. Deadline reminder: March 1, 2026")


if __name__ == "__main__":
    main()
