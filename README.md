# DAT200 CA3: Banana Quality Classification

**Group 37**
- Jannicke Ådalen
- Marcus Dalaker Figenschou
- Rikke Sellevold Vegstein

## Project Overview

This project implements machine learning models for binary classification of banana quality using Support Vector Machine with advanced feature selection techniques. The solution combines comprehensive data preprocessing, feature engineering with interaction terms, and hyperparameter optimization to achieve high classification accuracy.

**Final Result:** 1st place out of 66 teams on Kaggle (Score: 0.98906)

## Technical Approach

- **Model**: Support Vector Machine with RBF kernel
- **Feature Selection**: Sequential forward feature selection for optimal feature subset
- **Data Processing**: Z-score outlier removal, feature standardization, interaction term generation
- **Validation**: Multiple train/test splits with comprehensive hyperparameter tuning
- **Optimization**: Grid search with cross-validation for optimal model parameters

## Key Features

- Advanced outlier detection and removal using Z-score methodology
- Feature engineering with polynomial interaction terms
- Sequential forward selection for dimensionality reduction
- SVM with RBF kernel for non-linear classification
- Robust validation strategy with multiple data splits

## Results

- **Winner of DAT200 CA3 2025 Kaggle Competition**
- **1st place out of 66 teams**
- **Best accuracy: 98.906%**
- Demonstrated superior performance in banana quality prediction
- Successful application of feature selection and SVM optimization

## Files Structure

- `CA3.py` - Main analysis and model training script
- `CA3.ipynb` - Jupyter notebook version with detailed analysis
- `assets/` - Training and test datasets
- `results/` - Model predictions and submission files
- `pyproject.toml` - Project dependencies

## Requirements

See `pyproject.toml` for dependencies. Key requirements include scikit-learn, pandas, numpy, and matplotlib for comprehensive machine learning pipeline implementation.

## Usage

1. Install dependencies: `uv sync` or `pip install -e .`
2. Place dataset files in `assets/` directory
3. Run the main script: `python CA3.py`
4. Or use the Jupyter notebook: `jupyter notebook CA3.ipynb`

## Dataset

- **Source**: Banana quality measurements with physical and chemical properties
- **Target**: Binary classification (0 = Poor Quality, 1 = Good Quality)
- **Features**: Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity
- **Processing**: Feature engineering with interaction terms and standardization applied

## Competition

[Official Kaggle Competition](https://www.kaggle.com/competitions/dat-200-ca-3-2025)
