"""
run_pipeline.py — CLI entry point for the Penn Action pipeline.

Usage:
  python run_pipeline.py --extract        Run feature extraction for all actions
  python run_pipeline.py --train          Train batch models on extracted features
  python run_pipeline.py --evaluate       Evaluate trained batch models
  python run_pipeline.py --form-train     Train form evaluator (action + quality models)
  python run_pipeline.py --infer          Launch real-time webcam inference
  python run_pipeline.py --all            Run full pipeline (extract -> train -> evaluate -> form-train)
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
                        help='Train batch models on extracted features')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained batch models')
    parser.add_argument('--form-train', action='store_true', dest='form_train',
                        help='Train form evaluator (action classifier + quality regressor)')
    parser.add_argument('--infer', action='store_true',
                        help='Launch real-time webcam inference')
    parser.add_argument('--infer-classical', action='store_true', dest='infer_classical',
                        help='Launch the initial classical ML pose inference approach')
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline (extract -> train -> evaluate -> form-train)')
    parser.add_argument('--action', type=str, default=None,
                        help='Run extraction for a specific action only')

    args = parser.parse_args()

    # Default to --all if no flags
    if not any([args.extract, args.train, args.evaluate, args.form_train,
                args.infer, args.infer_classical, args.all]):
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

    # ── TRAIN (batch) ──
    results = None
    if args.train or args.all:
        from training import run_training_pipeline
        results = run_training_pipeline()

    # ── EVALUATE ──
    if args.evaluate or args.all:
        if results is None:
            from training import run_training_pipeline
            results = run_training_pipeline()

        from evaluation import run_evaluation
        metrics = run_evaluation(results)

        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        print(f"  Random Forest Accuracy: {metrics['rf_accuracy']*100:.2f}%")
        print(f"  SVM Accuracy:           {metrics['svm_accuracy']*100:.2f}%")

    # ── FORM-TRAIN ──
    if args.form_train or args.all:
        from form_evaluator_training import run_form_evaluator_pipeline
        run_form_evaluator_pipeline()

    # ── INFER ──
    if args.infer:
        from inference import main as inference_main
        inference_main()

    # ── INFER CLASSICAL (LEGACY) ──
    if args.infer_classical:
        from classical_inference import run_classical_inference
        run_classical_inference()

    elapsed = time.time() - start
    print(f"\nPipeline completed in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
