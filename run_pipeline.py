"""
run_pipeline.py — CLI entry point for the Penn Action pipeline.

Usage:
  python run_pipeline.py --extract       Run feature extraction for all actions
  python run_pipeline.py --train         Train models on extracted features
  python run_pipeline.py --evaluate      Evaluate trained models
  python run_pipeline.py --all           Run full pipeline (extract → train → evaluate)
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(
        description='Penn Action Classification Pipeline'
    )
    parser.add_argument('--extract', action='store_true',
                        help='Run feature extraction for all actions')
    parser.add_argument('--train', action='store_true',
                        help='Train models on extracted features')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained models')
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline')
    parser.add_argument('--action', type=str, default=None,
                        help='Run extraction for a specific action only')

    args = parser.parse_args()

    # Default to --all if no flags
    if not any([args.extract, args.train, args.evaluate, args.all]):
        args.all = True

    start = time.time()

    # ── EXTRACT ──
    if args.extract or args.all:
        from feature_extraction import extract_features_for_action, extract_all

        if args.action:
            print(f"\nExtracting features for: {args.action}")
            extract_features_for_action(args.action)
        else:
            print("\nExtracting features for all actions...")
            extract_all()

    # ── TRAIN ──
    results = None
    if args.train or args.all:
        from training import run_training_pipeline
        results = run_training_pipeline()

    # ── EVALUATE ──
    if args.evaluate or args.all:
        if results is None:
            # Need to run training first to get models
            from training import run_training_pipeline
            results = run_training_pipeline()

        from evaluation import run_evaluation
        metrics = run_evaluation(results)

        print("\n" + "="*60)
        print("  FINAL RESULTS")
        print("="*60)
        print(f"  Random Forest Accuracy: {metrics['rf_accuracy']*100:.2f}%")
        print(f"  SVM Accuracy:           {metrics['svm_accuracy']*100:.2f}%")

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
