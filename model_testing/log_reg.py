# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# %% Import Data

# Set working directory
if "ca3_banana_quality_classifier" in os.getcwd():
    os.chdir("..")  # Go up one level if we're in ca3_banana_quality_classifier

print(f"Working directory now: {os.getcwd()}")

# Load data
train_path = os.path.join("ca3_banana_quality_classifier", "assets", "train.csv")
test_path = os.path.join("ca3_banana_quality_classifier", "assets", "test.csv")

# Load data
# 1. Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# %% Chekcing for NaNs

print("---Train Data Info---")
train_df.info()

print("---Test Data Info---")
test_df.info()

# %% Remove unwanted features

train_df = train_df.drop(columns=["Peel Thickness", "Banana Density"], axis=1)
test_df = test_df.drop(columns=["Peel Thickness", "Banana Density"], axis=1)


# %% Split up data into X and y

X = train_df.drop(columns=["Quality"], axis=1)
y = train_df["Quality"]

# Scale features first - we'll apply scaling within each split later

# Define parameter ranges to test
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
degrees = [1, 2, 3, 4, 5]  # Polynomial degrees to test
random_seeds = [42, 123, 456, 789, 101112]  # Multiple random seeds

# Initialize results tracking
results = []

# Loop through different random seeds
for seed in random_seeds:
    print(f"\n----- Using random seed: {seed} -----")

    # Create train/validation split with current seed
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # Try different polynomial degrees
    for degree in degrees:
        print(f"  Testing polynomial degree: {degree}")

        # For degree > 2, limit the C values to avoid excessive overfitting
        current_C_values = C_values if degree <= 2 else [0.001, 0.01, 0.1, 1, 10]

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        # Scale features
        scaler = StandardScaler()
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_val_poly_scaled = scaler.transform(X_val_poly)

        # Try different C values with this degree
        for C in current_C_values:
            # Check if we might have too many features for higher degrees
            if X_train_poly.shape[1] > 1000:
                print(f"  Skipping C={C} for degree={degree} due to too many features ({X_train_poly.shape[1]})")
                continue

            # Train logistic regression
            lr = LogisticRegression(C=C, max_iter=2000, random_state=seed, solver="liblinear")

            try:
                lr.fit(X_train_poly_scaled, y_train)

                # Evaluate on validation set
                y_val_pred = lr.predict(X_val_poly_scaled)
                accuracy = accuracy_score(y_val, y_val_pred)

                # Store results
                results.append(
                    {
                        "seed": seed,
                        "degree": degree,
                        "C": C,
                        "accuracy": accuracy,
                        "n_features": X_train_poly.shape[1],
                    }
                )

                print(f"    C={C}, Validation Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"    Error with C={C}, degree={degree}: {e}")
                continue

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# Calculate average accuracy across seeds for each degree and C value
avg_results = results_df.groupby(["degree", "C"])["accuracy"].agg(["mean", "std", "count"]).reset_index()
avg_results = avg_results.sort_values("mean", ascending=False)

print("\n----- Top 10 Average Results Across Seeds -----")
print(avg_results.head(10))

# Identify best parameters based on highest average accuracy
best_params = avg_results.iloc[0]
best_degree = int(best_params["degree"])
best_C = best_params["C"]

print(f"\nBest parameters: degree={best_degree}, C={best_C}")
print(f"With average accuracy: {best_params['mean']:.4f}")

# Visualize results
plt.figure(figsize=(12, 8))
sns.lineplot(data=results_df, x="C", y="accuracy", hue="degree", marker="o", ci="sd")
plt.xscale("log")  # Log scale for C values
plt.title("Validation Accuracy vs. C Value for Different Polynomial Degrees")
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Validation Accuracy")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Heatmap of average accuracy by degree and C
pivot = avg_results.pivot(index="degree", columns="C", values="mean")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
plt.title("Average Validation Accuracy by Degree and C")
plt.tight_layout()

# Final model training with best parameters on full dataset
print("\n----- Training Final Model -----")

# Create polynomial features for full dataset
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(test_df)

# Scale features
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Train final model
final_model = LogisticRegression(C=best_C, max_iter=2000, random_state=42, solver="liblinear")
final_model.fit(X_poly_scaled, y)

print(f"Final model trained with degree={best_degree}, C={best_C}")
print(f"Number of features after polynomial transformation: {X_poly.shape[1]}")

# Make predictions on test set
test_predictions = final_model.predict(X_test_poly_scaled)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_predictions)), "Quality": test_predictions})
submission.to_csv(f"logreg_poly{best_degree}_C{best_C}_submission.csv", index=False)
print(f"Submission file created: logreg_poly{best_degree}_C{best_C}_submission.csv")
