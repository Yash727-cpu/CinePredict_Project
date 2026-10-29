import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# STEP 1: LOAD CLEANED DATASET
# =========================================================

df = pd.read_csv("dataset/cleaned_movies.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# STEP 2: SELECT ORIGINAL FEATURES
# =========================================================

features = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_year"
]

X = df[features]

y = df["success"]


print("\nFeatures used:")

for feature in features:
    print("-", feature)


# =========================================================
# STEP 3: DIVIDE DATASET
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# =========================================================
# STEP 4: CREATE DECISION TREE
# =========================================================

decision_tree = DecisionTreeClassifier(
    random_state=42
)


# =========================================================
# STEP 5: HYPERPARAMETER TUNING
# =========================================================

param_grid = {

    # How the tree chooses splits
    "criterion": [
        "gini",
        "entropy",
        "log_loss"
    ],

    # How the tree grows
    "splitter": [
        "best",
        "random"
    ],

    # Maximum tree depth
    "max_depth": [
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        15,
        20,
        None
    ],

    # Minimum samples required to split
    "min_samples_split": [
        2,
        5,
        10,
        20,
        30
    ],

    # Minimum samples at a leaf
    "min_samples_leaf": [
        1,
        2,
        5,
        10,
        15,
        20
    ],

    # Number of features considered during split
    "max_features": [
        None,
        "sqrt",
        "log2"
    ],

    # Cost-complexity pruning
    "ccp_alpha": [
        0.0,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01
    ],

    # Class weighting
    "class_weight": [
        None,
        "balanced"
    ]
}


print("\n========================================")
print("   ADVANCED DECISION TREE TUNING")
print("========================================")

print("\nTesting multiple configurations...")
print("This may take some time.")
print("Please wait...")


# =========================================================
# STEP 6: GRID SEARCH
# =========================================================

grid_search = RandomizedSearchCV(
    estimator=decision_tree,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

grid_search.fit(X_train, y_train)


grid_search.fit(
    X_train,
    y_train
)


# =========================================================
# STEP 7: DISPLAY BEST PARAMETERS
# =========================================================

print("\n========================================")
print("       BEST PARAMETERS")
print("========================================")

print("\nBest parameters:")

print(grid_search.best_params_)


print("\nBest cross-validation accuracy:")

print(
    f"{grid_search.best_score_ * 100:.2f}%"
)


# =========================================================
# STEP 8: GET BEST MODEL
# =========================================================

model = grid_search.best_estimator_

print("\nBest Decision Tree selected successfully!")


# =========================================================
# STEP 9: TRAIN BEST MODEL
# =========================================================

model.fit(
    X_train,
    y_train
)

print("Best model trained successfully!")


# =========================================================
# STEP 10: SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "model/movie_success_model.pkl"
)

print("Model saved successfully!")


# =========================================================
# STEP 11: TEST MODEL
# =========================================================

y_pred = model.predict(X_test)

print("\nModel testing completed!")


# =========================================================
# STEP 12: CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# =========================================================
# STEP 13: DISPLAY FINAL RESULTS
# =========================================================

print("\n========================================")
print("       FINAL MODEL EVALUATION")
print("========================================")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1-Score : {f1 * 100:.2f}%"
)


# =========================================================
# STEP 14: CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


print("\nConfusion Matrix Explanation:")

print(
    "True Negative :",
    cm[0][0]
)

print(
    "False Positive:",
    cm[0][1]
)

print(
    "False Negative:",
    cm[1][0]
)

print(
    "True Positive  :",
    cm[1][1]
)


# =========================================================
# STEP 15: DISPLAY CONFUSION MATRIX
# =========================================================

# STEP 15: DISPLAY AND SAVE CONFUSION MATRIX

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Successful", "Successful"],
    yticklabels=["Not Successful", "Successful"],
    linewidths=1,
    cbar=True
)

plt.title("CinePredict - Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

# Save confusion matrix as PNG
output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(exist_ok=True)

plt.savefig(
    output_dir / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Confusion matrix saved successfully!")

# Display graphical confusion matrix
plt.show()
