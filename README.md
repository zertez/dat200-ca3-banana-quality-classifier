# Banana Quality Classifier

**Winner of [DAT-200 CA-3 2025 Kaggle Competition](https://www.kaggle.com/competitions/dat-200-ca-3-2025) - 1st Place out of 66 Teams**

Machine learning project that classifies banana quality using Support Vector Machine with feature selection, achieving 98.9% accuracy.

## Results

- **1st Place** in Kaggle competition (66 teams)
- **Best Score**: 0.98906
- **Algorithm**: SVM with RBF kernel and sequential feature selection

## Team (Group 37)

- Jannicke Ådalen
- Marcus Dalaker Figenschou (Kaggle: zertez)
- Rikke Sellevold Vegstein

## Dataset

Banana quality classification with features: Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity → Quality (0/1)

## Key Features

- **Data Processing**: Z-score outlier removal, feature engineering with interaction terms
- **Model**: SVM with RBF kernel, sequential forward feature selection
- **Validation**: Multiple train/test splits with hyperparameter tuning

## Usage

1. Install dependencies: `uv sync` or `pip install -e .`
2. Place dataset files in `assets/` directory
3. Run: `python CA3.py`

## Files

- `CA3.py` - Main analysis script
- `CA3.ipynb` - Jupyter notebook version
- `pyproject.toml` - Dependencies

## Competition

[Official Kaggle Competition](https://www.kaggle.com/competitions/dat-200-ca-3-2025)