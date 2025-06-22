#!/bin/bash
# Batch script for running synthetic flower generation

set -e  # Exit on any error

echo "Starting synthetic flower generation..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Check CPU count
CPU_COUNT=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())")
echo "Available CPUs: $CPU_COUNT"

if [ "$CPU_COUNT" -lt 40 ]; then
    echo "Warning: Only $CPU_COUNT CPUs available. Recommend 40+ for optimal performance."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Check if data directories exist
if [ ! -d "../data/petals" ]; then
    echo "Error: ../data/petals directory not found"
    echo "Please ensure your petal images are organized as:"
    echo "  ../data/petals/{color}/*.png"
    echo "  ../data/petals/{color}/crown/*.png"
    exit 1
fi

echo "Data directory structure:"
ls -la ../data/petals/ || echo "Could not list petals directory"

# Create output directories
mkdir -p ../data/synthetic_flw/{flw,mask}
echo "Created output directories"

# Run the synthesis
echo "Running synthesis script..."
python create_synthetic.py

echo "Synthesis completed successfully!"
echo "Output saved to: ../data/synthetic_flw/"