#!/bin/bash
# BERT Training Runner
# Quick script to train BERT on the augmented TREC dataset
# Usage: ./run_bert.sh [options]

# Default arguments
ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            ARGS="$ARGS --epochs $2"
            shift 2
            ;;
        --batch-size)
            ARGS="$ARGS --batch-size $2"
            shift 2
            ;;
        --runs)
            ARGS="$ARGS --runs $2"
            shift 2
            ;;
        --compare)
            ARGS="$ARGS --compare"
            shift
            ;;
        --save-model)
            ARGS="$ARGS --save-model"
            shift
            ;;
        --help)
            echo "Usage: ./run_bert.sh [options]"
            echo ""
            echo "Options:"
            echo "  --epochs N          Number of training epochs (default: 3)"
            echo "  --batch-size N      Batch size (default: 16)"
            echo "  --runs N            Number of runs with different seeds (default: 1)"
            echo "  --compare           Compare multiple configurations"
            echo "  --save-model        Save the trained model"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_bert.sh                    # Run with default settings"
            echo "  ./run_bert.sh --epochs 5         # Train for 5 epochs"
            echo "  ./run_bert.sh --runs 3           # Run 3 times with different seeds"
            echo "  ./run_bert.sh --compare          # Compare multiple configurations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run BERT training
echo "Running BERT training..."
uv run python -m bert_pipeline.run_bert $ARGS
