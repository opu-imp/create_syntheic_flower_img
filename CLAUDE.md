# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for generating synthetic flower images using computer vision techniques. The core functionality synthesizes flower images by combining individual petal images arranged in spiral patterns that mimic natural flower growth.

## Architecture

The project follows a modular design with these key components:

### Core Synthesis Pipeline
- **create_synthe_v4.py**: Main synthesis script that generates synthetic flower images by arranging petals in spiral patterns
- **SynthesisParameterConfig**: Main configuration class that manages petal images, crown images, and angle calculations
- **multi_channel_img_io.py**: Handles multi-channel mask I/O using HDF5 format

### Image Processing Utilities
- **utils.py**: Computer vision utilities including IoU calculation, NMS (Non-Maximum Suppression), and bbox sorting functions
- Geometric transformations for petal rotation and positioning
- Edge refinement and foreground cropping functions

### Matching and Analysis
- **match.py**, **match_arrange.py**: Pattern matching algorithms for flower analysis
- **calculate_match.py**: Distance calculation utilities
- **arrange_bbox.py**: Bounding box arrangement functions
- **generate_ground_truth.py**: Ground truth data generation

## Key Data Flow

1. **Input**: Individual petal images stored in `../data/petals/{color}/` with crown images in `../data/petals/{color}/crown/`
2. **Processing**: Petals are arranged using spiral angle calculations mimicking natural phyllotaxis (golden angle spirals)
3. **Output**: Synthetic flower images and multi-channel masks saved to `../data/synthetic_flw/`

## Development Commands

### Docker Environment
```bash
# Build container
docker build -t synthetic-flowers .

# Run container
docker run -v $(pwd):/work -p 8899:8899 synthetic-flowers
```

### Jupyter Lab
```bash
# Start Jupyter Lab server
./start-jupyter.sh
# Access at localhost:8899 with token 'ttt'
```

### Python Execution
```bash
# Run main synthesis script (requires 40+ CPU cores)
python create_synthe_v4.py

# Process individual color/batch
python -c "from create_synthe_v4 import main; main((color_idx, batch_idx, num_create))"
```

## Data Requirements

The project expects petal images to be organized as:
```
../data/petals/{color}/*.png          # Individual petal images  
../data/petals/{color}/crown/*.png    # Crown/center images
```

Colors supported: '黄色丸', '紫', '白紫', '薄い白緑', '薄黄色'

## Performance Notes

- Main synthesis script uses multiprocessing and requires 40+ CPU cores
- Generates 12,500 images per color/batch combination
- Uses ProcessPoolExecutor for parallel processing across 5 colors × 8 batches
- HDF5 format used for efficient multi-channel mask storage

## Spiral Pattern Generation

The core algorithm uses phyllotaxis principles:
- Base angle of 137° (golden angle) for natural spiral patterns
- Configurable petal counts and arrangements via `dic_pairs` parameter
- Noise injection for realistic variation in petal positioning