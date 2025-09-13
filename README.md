# Banana Quality Classifier

**Winner of Kaggle Competition - 1st Place out of 66 Teams**

A machine learning project that classifies banana quality using various physical characteristics. This project uses Support Vector Machine (SVM) with feature selection to achieve high accuracy in distinguishing between good and bad quality bananas.

**Competition Results**: Achieved 1st place in the [DAT-200 CA-3 2025 Kaggle Competition](https://www.kaggle.com/competitions/dat-200-ca-3-2025) with a score of 0.98906, competing against 65 other teams.

## Project Overview

This project was developed as part of the DAT-200 machine learning course assignment (CA3) and demonstrates:
- Data exploration and visualization
- Outlier detection and removal
- Feature engineering with interaction terms
- Sequential feature selection
- SVM hyperparameter tuning
- Model evaluation and validation
- Competitive machine learning methodology

**Competition**: [DAT-200 CA-3 2025](https://www.kaggle.com/competitions/dat-200-ca-3-2025) on Kaggle

**Team Members (Group 37):**
- Jannicke Ådalen
- Marcus Dalaker Figenschou (Kaggle: zertez)
- Rikke Sellevold Vegstein

## Dataset

The dataset contains banana quality measurements with the following features:
- **Size**: Physical size of the banana
- **Weight**: Weight in grams
- **Sweetness**: Sweetness level
- **Softness**: Texture measurement
- **HarvestTime**: Time since harvest
- **Ripeness**: Ripeness level
- **Acidity**: Acidity measurement
- **Quality**: Target variable (0 = bad quality, 1 = good quality)

**Note:** The original dataset included "Peel Thickness" and "Banana Density" features that were removed as specified in the assignment requirements.

## Key Features

### Data Processing
- **Outlier Removal**: Z-score based outlier detection (threshold = 3)
- **Feature Engineering**: Created interaction terms based on correlation analysis
- **Data Standardization**: StandardScaler for feature normalization

### Machine Learning Pipeline
- **Algorithm**: Support Vector Machine with RBF kernel
- **Feature Selection**: Sequential Forward Selection with automatic stopping
- **Validation**: Multiple train/test splits for robust evaluation
- **Hyperparameter Tuning**: Grid search over C and gamma parameters

### Model Performance
- **Competition Winning Score**: 0.98906 (1st place out of 66 teams)
- **Reproduced Score**: 0.98541
- **Selected Features**: Automatically determined through sequential selection
- **Algorithm**: Support Vector Machine with RBF kernel and sequential feature selection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ca3_banana_quality_classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv sync
```

Or if you're not using uv:
```bash
pip install -e .
```

## Usage

### Data Setup
1. Download the dataset files and place them in the `assets/` directory:
   - `train.csv` - Training data
   - `test.csv` - Test data for predictions

### Running the Analysis
```bash
python CA3.py
```

This will:
1. Load and explore the data
2. Remove outliers and engineer features
3. Perform feature selection and hyperparameter tuning
4. Train the final model
5. Generate predictions for the test set
6. Save results to a CSV file

### Output Files
- `submission_svc_k<kernel>_C<C>_g<gamma>.csv` - Kaggle submission file
- Various visualization plots during execution

## Project Structure

```
ca3_banana_quality_classifier/
├── CA3.py                 # Main analysis script
├── CA3.ipynb             # Jupyter notebook version
├── README.md             # This file
├── pyproject.toml        # Python dependencies and project config
├── assets/               # Data files (not included in repo)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── .gitignore           # Git ignore rules
└── .venv/               # Virtual environment
```

## Methodology

### 1. Data Exploration
- **Visualization**: Boxplots and histograms to understand feature distributions
- **Quality Analysis**: Separate analysis for each quality class
- **Outlier Detection**: Identified extreme values using statistical methods

### 2. Data Cleaning
- **Outlier Removal**: Z-score > 3 threshold applied per quality group
- **Feature Removal**: Dropped specified unwanted features
- **Data Integrity**: Maintained stratification across quality classes

### 3. Feature Engineering
- **Correlation Analysis**: Heatmap to identify feature relationships
- **Interaction Terms**: Created 6 interaction features:
  - Weight × Sweetness
  - Size × HarvestTime
  - Sweetness × HarvestTime
  - Size × Sweetness
  - Weight × Acidity
  - Ripeness × Acidity

### 4. Model Development
- **Algorithm Selection**: Tested multiple algorithms, settled on SVM
- **Feature Selection**: Sequential Forward Selection with 0.1% tolerance
- **Hyperparameter Tuning**: Grid search with multiple validation splits
- **Model Validation**: 10 different train/test splits for robust evaluation

## Results and Insights

### Key Findings
1. **Bimodal Distributions**: Weight and softness showed clear separation between quality classes
2. **Feature Importance**: Size and weight emerged as strong quality indicators
3. **Outlier Impact**: Bad quality bananas had more extreme feature values
4. **Model Performance**: SVM with RBF kernel achieved excellent classification accuracy

### Competition Winning Model Configuration
- **Kernel**: RBF (Support Vector Machine)
- **Optimal Parameters**: Determined through extensive grid search
- **Selected Features**: Automatically chosen via sequential forward selection
- **Final Competition Score**: 0.98906 (1st place out of 66 teams)
- **Validation Score**: >98% accuracy across multiple splits

## Future Improvements

1. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
2. **Model Comparison**: Add comparison with other algorithms (Random Forest, XGBoost)
3. **Feature Analysis**: Deeper analysis of feature importance and selection
4. **Code Modularization**: Split into separate modules for data processing, modeling, and visualization
5. **Deployment**: Create a web interface for real-time predictions

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Competition Link

**Official Competition**: [DAT-200 CA-3 2025 on Kaggle](https://www.kaggle.com/competitions/dat-200-ca-3-2025)

## License

This project is for educational purposes as part of a machine learning course assignment.

## Contact

For questions about this project, please contact the team members through the course platform.
