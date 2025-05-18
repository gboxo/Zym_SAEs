# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import pandas as pd

import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from collections import defaultdict


path = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/latent_scoring/latent_scoring_0_new/features/"

with open(os.path.join(path, "features_M0_D0.pkl"), "rb") as f:
    features = pkl.load(f)



#mean_features = {key:np.array(feat.todense()).sum(axis=0) for key, feat in features.items()}
with open(os.path.join(path, "mean_features_M0_D0.pkl"), "rb") as f:
    mean_features = pkl.load(f)


# %%
df_path = os.path.join("/users/nferruz/gboxo/finetune_SAE_DMS/alpha-amylase-training-data.csv")
df = pd.read_csv(df_path)


activity_dict = {key:df[df["mutant"] == key]["activity_dp7"].values[0] for key in mean_features.keys()}


random_indices = np.random.permutation(len(mean_features))
train_indices = random_indices[:int(len(random_indices)*0.8)]
test_indices = random_indices[int(len(random_indices)*0.8):]

all_keys = list(mean_features.keys())
train_keys = [all_keys[i] for i in train_indices]
test_keys = [all_keys[i] for i in test_indices]

train_features = {key:mean_features[key] for key in train_keys}
test_features = {key:mean_features[key] for key in test_keys}

train_activity = {key:activity_dict[key] for key in train_keys}
test_activity = {key:activity_dict[key] for key in test_keys}



train_features_arr = np.array([train_features[key] for key in train_keys])
test_features_arr = np.array([test_features[key] for key in test_keys])
train_activity_arr = np.array([train_activity[key] for key in train_keys])
test_activity_arr = np.array([test_activity[key] for key in test_keys])


# %%


"""
FOR ABLATION: 


I want to find which featuers don't fire for the top-x% of the sequences in terms of activity.
"""

def get_top_x_percent_features(features_arr, activity_arr, min_activity=2, min_rest_fraction=0.01):
    """
    Get the features that don't fire for sequences with activty higher than min_activity,
    but do fire on at least min_rest_fraction of the remaining sequences.
    
    Args:
        features_arr: Array of features for each sequence
        activity_arr: Array of activity values for each sequence
        min_activity: Minimum activity value for sequences to be considered
        min_rest_fraction: Minimum fraction of remaining sequences that must have the feature (default 0.1)
    
    Returns:
        Array of indices for features that meet the criteria
    """
    # Split into top x% and rest
    top_mask = activity_arr >= min_activity
    rest_mask = ~top_mask
    
    # Get features for each group
    top_features = features_arr[top_mask]
    rest_features = features_arr[rest_mask]
    
    # Find features that don't fire in top x%
    top_zero = np.all(top_features == 0, axis=0)
    
    # Find features that fire in at least min_rest_fraction of rest
    rest_firing = np.sum(rest_features > 0, axis=0) / rest_features.shape[0]
    rest_sufficient = rest_firing >= min_rest_fraction
    
    # Return indices where both conditions are met
    return np.where(top_zero & rest_sufficient)[0]




# %%
all_features = []
for min_activity in np.arange(1, 10, 0.1):
    features = get_top_x_percent_features(train_features_arr, train_activity_arr, min_activity = min_activity, min_rest_fraction=0.1)
    all_features.append(features)


# %%
import seaborn as sns
# Get all unique features across all arrays
unique_features = np.unique(np.concatenate([arr for arr in all_features if arr.size > 0]))

# Create a binary matrix: rows = features, columns = min_activity thresholds
# 1 if feature is present at that threshold, 0 otherwise
feature_matrix = np.zeros((len(unique_features), len(all_features)))

# Fill the matrix
for i, feature_arr in enumerate(all_features):
    for feature in feature_arr:
        # Find the index of this feature in unique_features
        feature_idx = np.where(unique_features == feature)[0][0]
        feature_matrix[feature_idx, i] = 1

# Create activity thresholds for x-axis labels (from 1 to 10 with 0.1 increment)
activity_thresholds = np.arange(1, 10, 0.1)

# Plot the heatmap with transposed matrix (features on y-axis, activity on x-axis)
plt.figure(figsize=(16, 12))
ax = sns.heatmap(feature_matrix, cmap="Blues", 
                 yticklabels=unique_features,
                 xticklabels=[f"{a:.1f}" for a in activity_thresholds], 
                 cbar_kws={'label': 'Feature Present'})

# Adjust x-axis labels to be readable (vertical and show every nth label)
n = 5  # Show every 5th activity threshold label
for idx, label in enumerate(ax.get_xticklabels()):
    if idx % n != 0:
        label.set_visible(False)
    else:
        label.set_rotation(90)

plt.title('Feature Presence at Different Minimum Activity Thresholds')
plt.xlabel('Minimum Activity Threshold')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.show()


# %%


def find_optimal_threshold(test_features, feature_indices, test_activity, activity_threshold):
    """
    Find optimal classification threshold that maximizes balanced accuracy
    """
    # For this classifier, we're checking if ALL features are absent
    # We can't adjust probability threshold, but we can adjust how many features need to be absent
    
    # One approach: require only a percentage of features to be absent
    results = []
    feature_counts = np.sum(test_features[:, feature_indices] == 0, axis=1)
    max_features = len(feature_indices)
    
    # Try different thresholds for how many features need to be absent
    for percent_threshold in np.linspace(0.5, 1.0, 11):  # Try 50% to 100% of features
        feature_threshold = int(max_features * percent_threshold)
        predictions = feature_counts >= feature_threshold
        
        metrics = evaluate_classifier_balanced(predictions, test_activity, activity_threshold)
        metrics["percent_threshold"] = percent_threshold
        results.append(metrics)
    
    # Find threshold with best balanced accuracy
    results_df = pd.DataFrame(results)
    best_idx = results_df["balanced_accuracy"].idxmax()
    
    return results_df.iloc[best_idx]["percent_threshold"], results_df.iloc[best_idx]




# Define a function to predict activity based on feature absence
def predict_activity_by_feature_absence(test_features, feature_indices, activity_threshold):
    """
    Predict whether activity is above/below threshold by checking absence of specific features.
    
    Args:
        test_features: Array of features for test sequences
        feature_indices: Array of feature indices that don't fire for high-activity sequences
        activity_threshold: Threshold to classify activity as high/low
    
    Returns:
        predictions: Binary array (True = high activity, False = low activity)
    """
    # Check if the selected features are absent (= 0) in test sequences
    # If all selected features are absent, predict high activity; otherwise, predict low activity
    feature_absence = np.all(test_features[:, feature_indices] == 0, axis=1)
    return feature_absence  # True means predicted high activity (above threshold)

# Function to evaluate classifier performance
def evaluate_classifier_balanced(predictions, ground_truth, threshold):
    """
    Evaluate classifier with metrics suitable for imbalanced data
    """
    actual_classes = ground_truth >= threshold
    
    # Calculate metrics
    true_positives = np.sum((predictions == True) & (actual_classes == True))
    false_positives = np.sum((predictions == True) & (actual_classes == False))
    true_negatives = np.sum((predictions == False) & (actual_classes == False))
    false_negatives = np.sum((predictions == False) & (actual_classes == True))
    
    # Standard accuracy
    accuracy = (true_positives + true_negatives) / len(predictions)
    
    # Class-specific metrics
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Balanced accuracy (average of sensitivity and specificity)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": sensitivity,  # Recall is the same as sensitivity
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "pos_count": true_positives + false_negatives,  # Total positive examples
        "neg_count": true_negatives + false_positives   # Total negative examples
    }

# Test the classifier with adjustments for class imbalance
activity_thresholds = np.arange(1, 10, 0.5)
balanced_results = []

for threshold in activity_thresholds:
    # Get feature indices for this threshold
    threshold_idx = int((threshold - 1) * 10)
    threshold_idx = min(threshold_idx, len(all_features) - 1)
    
    feature_indices = all_features[threshold_idx]
    
    # Skip if no features were found
    if len(feature_indices) == 0:
        continue
    
    # Check class balance for this threshold
    actual_classes = test_activity_arr >= threshold
    pos_count = np.sum(actual_classes)
    neg_count = len(actual_classes) - pos_count
    class_ratio = pos_count / len(actual_classes)
    
    # Find optimal decision threshold
    opt_threshold, opt_metrics = find_optimal_threshold(
        test_features_arr, feature_indices, test_activity_arr, threshold)
    
    # Store results
    result = {
        "activity_threshold": threshold,
        "num_features": len(feature_indices),
        "class_ratio": class_ratio,
        "optimal_percent_threshold": opt_threshold,
        **opt_metrics
    }
    balanced_results.append(result)

# Convert to DataFrame
balanced_df = pd.DataFrame(balanced_results)

# Display key results
print(balanced_df[["activity_threshold", "class_ratio", "optimal_percent_threshold", 
                 "balanced_accuracy", "sensitivity", "specificity", "f1"]])

# Plot balanced metrics
plt.figure(figsize=(12, 8))
for metric in ["balanced_accuracy", "sensitivity", "specificity", "f1"]:
    plt.plot(balanced_df["activity_threshold"], balanced_df[metric], marker='o', label=metric)

plt.xlabel("Activity Threshold")
plt.ylabel("Score")
plt.title("Balanced Classifier Performance at Different Activity Thresholds")
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label="Random Guess")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot class distribution
plt.figure(figsize=(10, 6))
plt.bar(balanced_df["activity_threshold"], balanced_df["class_ratio"], color='skyblue')
plt.axhline(y=0.5, color='r', linestyle='--', label="Balanced (50%)")
plt.xlabel("Activity Threshold")
plt.ylabel("Fraction of Positive Class")
plt.title("Class Balance at Different Activity Thresholds")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find best performing threshold by balanced accuracy
best_idx = balanced_df["balanced_accuracy"].idxmax()
best_threshold = balanced_df.iloc[best_idx]["activity_threshold"]
best_opt_threshold = balanced_df.iloc[best_idx]["optimal_percent_threshold"]

print(f"Best performing threshold: {best_threshold} with balanced accuracy: {balanced_df.iloc[best_idx]['balanced_accuracy']:.4f}")
print(f"Optimal percent of features required: {best_opt_threshold:.2f}")

# Show confusion matrix for best threshold
threshold_idx = int((best_threshold - 1) * 10)
feature_indices = all_features[threshold_idx]

# Apply optimal threshold for feature absence percentage
feature_counts = np.sum(test_features_arr[:, feature_indices] == 0, axis=1)
max_features = len(feature_indices)
feature_threshold = int(max_features * best_opt_threshold)
predictions = feature_counts >= feature_threshold

actual_classes = test_activity_arr >= best_threshold

conf_matrix = np.array([
    [np.sum((predictions == False) & (actual_classes == False)), 
     np.sum((predictions == False) & (actual_classes == True))],
    [np.sum((predictions == True) & (actual_classes == False)), 
     np.sum((predictions == True) & (actual_classes == True))]
])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Low Activity", "High Activity"],
            yticklabels=["Predicted Low", "Predicted High"])
plt.title(f"Confusion Matrix (Activity Threshold = {best_threshold}, Feature % = {best_opt_threshold:.2f})")
plt.ylabel("Predicted Label")
plt.xlabel("True Label")
plt.tight_layout()
plt.show()

# Find optimal threshold for each activity level
def find_optimal_threshold(test_features, feature_indices, test_activity, activity_threshold):
    """
    Find optimal classification threshold that maximizes balanced accuracy
    """
    # For this classifier, we're checking if ALL features are absent
    # We can't adjust probability threshold, but we can adjust how many features need to be absent
    
    # One approach: require only a percentage of features to be absent
    results = []
    feature_counts = np.sum(test_features[:, feature_indices] == 0, axis=1)
    max_features = len(feature_indices)
    
    # Try different thresholds for how many features need to be absent
    for percent_threshold in np.linspace(0.5, 1.0, 11):  # Try 50% to 100% of features
        feature_threshold = int(max_features * percent_threshold)
        predictions = feature_counts >= feature_threshold
        
        metrics = evaluate_classifier_balanced(predictions, test_activity, activity_threshold)
        metrics["percent_threshold"] = percent_threshold
        results.append(metrics)
    
    # Find threshold with best balanced accuracy
    results_df = pd.DataFrame(results)
    best_idx = results_df["balanced_accuracy"].idxmax()
    
    return results_df.iloc[best_idx]["percent_threshold"], results_df.iloc[best_idx]

# %%
# First, let's define the grid search parameters
min_rest_fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
activity_thresholds = np.arange(1, 10, 0.5)
n_folds = 5

# %%
# Implement cross-validation for feature selection and evaluation
def cross_validate_feature_selection(features_arr, activity_arr, min_activity, min_rest_fraction, n_folds=5):
    """
    Perform cross-validation for feature selection and evaluation
    
    Args:
        features_arr: Array of features
        activity_arr: Array of activity values
        min_activity: Minimum activity threshold
        min_rest_fraction: Minimum fraction of remaining sequences with feature
        n_folds: Number of CV folds
        
    Returns:
        Dictionary with cross-validation results
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []
    
    # For each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(features_arr)):
        # Split data into train and test
        X_train, X_test = features_arr[train_idx], features_arr[test_idx]
        y_train, y_test = activity_arr[train_idx], activity_arr[test_idx]
        
        # Select features using training data only
        selected_features = get_top_x_percent_features(
            X_train, y_train, min_activity=min_activity, min_rest_fraction=min_rest_fraction)
        
        # If no features were found, skip this fold
        if len(selected_features) == 0:
            continue
        
        # Evaluate on test data
        feature_counts = np.sum(X_test[:, selected_features] == 0, axis=1)
        max_features = len(selected_features)
        
        # Try different thresholds for feature absence percentage
        best_balanced_acc = 0
        best_pct_threshold = 0
        best_metrics = None
        
        for pct_threshold in np.linspace(0.5, 1.0, 11):
            feature_threshold = int(max_features * pct_threshold)
            predictions = feature_counts >= feature_threshold
            
            metrics = evaluate_classifier_balanced(predictions, y_test, min_activity)
            
            if metrics["balanced_accuracy"] > best_balanced_acc:
                best_balanced_acc = metrics["balanced_accuracy"]
                best_pct_threshold = pct_threshold
                best_metrics = metrics
        
        # Add fold results
        if best_metrics is not None:
            fold_result = {
                "fold": fold,
                "num_features": len(selected_features),
                "feature_indices": selected_features,
                "percent_threshold": best_pct_threshold,
                **best_metrics
            }
            cv_results.append(fold_result)
    
    # Aggregate results across folds
    if not cv_results:
        return {"valid": False}
    
    # Calculate mean and std of performance metrics
    metrics_to_agg = ["balanced_accuracy", "sensitivity", "specificity", "f1", "accuracy"]
    agg_results = {
        "min_activity": min_activity,
        "min_rest_fraction": min_rest_fraction,
        "valid": True,
        "num_folds": len(cv_results),
        "avg_num_features": np.mean([r["num_features"] for r in cv_results]),
        "avg_percent_threshold": np.mean([r["percent_threshold"] for r in cv_results])
    }
    
    for metric in metrics_to_agg:
        values = [r[metric] for r in cv_results]
        agg_results[f"avg_{metric}"] = np.mean(values)
        agg_results[f"std_{metric}"] = np.std(values)
    
    return agg_results

# %%
# Run grid search with CV
grid_results = []

for min_rest_fraction in min_rest_fractions:
    for min_activity in activity_thresholds:
        print(f"Evaluating min_rest_fraction={min_rest_fraction}, min_activity={min_activity}")
        
        cv_result = cross_validate_feature_selection(
            train_features_arr, train_activity_arr, 
            min_activity=min_activity, 
            min_rest_fraction=min_rest_fraction,
            n_folds=n_folds
        )
        
        if cv_result["valid"]:
            cv_result["min_activity"] = min_activity
            cv_result["min_rest_fraction"] = min_rest_fraction
            grid_results.append(cv_result)

# Convert to DataFrame
grid_df = pd.DataFrame(grid_results)

# %%
# Plot heatmap of balanced accuracy across parameter grid
if len(grid_df) > 0:
    # Pivot the data for heatmap
    heatmap_data = grid_df.pivot(
        index="min_rest_fraction", 
        columns="min_activity", 
        values="avg_balanced_accuracy"
    )
    
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", 
                    cbar_kws={'label': 'Average Balanced Accuracy'})
    plt.title('Grid Search Results: Balanced Accuracy by Parameter Combination')
    plt.xlabel('Activity Threshold')
    plt.ylabel('Minimum Rest Fraction')
    plt.tight_layout()
    plt.show()
    
    # Find best parameter combination
    best_idx = grid_df["avg_balanced_accuracy"].idxmax()
    best_params = grid_df.iloc[best_idx]
    
    print("Best parameter combination:")
    print(f"min_activity = {best_params['min_activity']}")
    print(f"min_rest_fraction = {best_params['min_rest_fraction']}")
    print(f"Average balanced accuracy: {best_params['avg_balanced_accuracy']:.4f} ± {best_params['std_balanced_accuracy']:.4f}")
    print(f"Average F1 score: {best_params['avg_f1']:.4f} ± {best_params['std_f1']:.4f}")
    print(f"Average number of features: {best_params['avg_num_features']:.1f}")
    print(f"Average percent threshold: {best_params['avg_percent_threshold']:.2f}")
else:
    print("No valid results found in grid search.")

# %%
# Now apply the best parameters to get the final set of features
if len(grid_df) > 0:
    best_min_activity = best_params['min_activity']
    best_min_rest_fraction = best_params['min_rest_fraction']
    
    # Get features with best parameters using all training data
    final_features = get_top_x_percent_features(
        train_features_arr, train_activity_arr, 
        min_activity=best_min_activity, 
        min_rest_fraction=best_min_rest_fraction
    )
    
    print(f"Selected {len(final_features)} features using best parameters")
    print(f"Feature indices: {final_features}")
    
    # Final evaluation on test set
    if len(final_features) > 0:
        # Find optimal percentage threshold on training data
        feature_counts_train = np.sum(train_features_arr[:, final_features] == 0, axis=1)
        max_features = len(final_features)
        
        best_balanced_acc = 0
        best_pct_threshold = 0
        
        for pct_threshold in np.linspace(0.5, 1.0, 21):  # More fine-grained search
            feature_threshold = int(max_features * pct_threshold)
            predictions = feature_counts_train >= feature_threshold
            
            metrics = evaluate_classifier_balanced(predictions, train_activity_arr, best_min_activity)
            
            if metrics["balanced_accuracy"] > best_balanced_acc:
                best_balanced_acc = metrics["balanced_accuracy"]
                best_pct_threshold = pct_threshold
        
        print(f"Optimal percentage threshold on training data: {best_pct_threshold:.3f}")
        
        # Evaluate on test set
        feature_counts_test = np.sum(test_features_arr[:, final_features] == 0, axis=1)
        feature_threshold = int(max_features * best_pct_threshold)
        test_predictions = feature_counts_test >= feature_threshold
        
        test_metrics = evaluate_classifier_balanced(test_predictions, test_activity_arr, best_min_activity)
        
        print("\nTest set performance:")
        print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"Specificity: {test_metrics['specificity']:.4f}")
        print(f"F1 score: {test_metrics['f1']:.4f}")
        
        # Confusion matrix
        actual_classes = test_activity_arr >= best_min_activity
        
        conf_matrix = np.array([
            [test_metrics["true_negatives"], test_metrics["false_negatives"]],
            [test_metrics["false_positives"], test_metrics["true_positives"]]
        ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
                    xticklabels=["Low Activity", "High Activity"],
                    yticklabels=["Predicted Low", "Predicted High"])
        plt.title(f"Confusion Matrix (Optimal Parameters)")
        plt.ylabel("Predicted Label")
        plt.xlabel("True Label")
        plt.tight_layout()
        plt.show()
        
        # Feature importance analysis
        if len(final_features) > 0:
            # For each feature, calculate its individual predictive power
            feature_importance = []
            
            for i, feature_idx in enumerate(final_features):
                # Evaluate using just this feature
                single_feature_absence = test_features_arr[:, feature_idx] == 0
                metrics = evaluate_classifier_balanced(single_feature_absence, test_activity_arr, best_min_activity)
                
                feature_importance.append({
                    "feature_idx": feature_idx,
                    "balanced_acc": metrics["balanced_accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "f1": metrics["f1"]
                })
            
            # Convert to DataFrame and sort by importance
            importance_df = pd.DataFrame(feature_importance)
            importance_df = importance_df.sort_values("balanced_acc", ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.bar(importance_df["feature_idx"].astype(str), importance_df["balanced_acc"])
            plt.axhline(y=0.5, color='r', linestyle='--', label="Random Guess")
            plt.xlabel("Feature Index")
            plt.ylabel("Balanced Accuracy")
            plt.title("Individual Feature Importance")
            plt.xticks(rotation=90)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print("\nTop 5 most important features:")
            print(importance_df.head(5))



"""
Differentiate sequences with >4 activity


Top 5 most important features:
     feature_idx  balanced_acc  sensitivity  specificity        f1
514        10979      0.579799          1.0     0.159598  0.019531
315         7165      0.568917          1.0     0.137835  0.019048
389         8471      0.565011          1.0     0.130022  0.018880
473        10170      0.554129          1.0     0.108259  0.018428
444         9612      0.553850          1.0     0.107701  0.018416

"""


# %%




"""

FOR STEERING: 


I want to find which features are most important for detecting with activity higer than min_activity



THINGS TO TEST:
- difference of means between high and low activity
- sparse probes
- sparse PCA
"""


#  %%

# Difference of means between high and low activity


def compute_metrics(preds, y_test):
    y_pred = preds

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred)   
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Compute classification report
    class_report = classification_report(y_test, y_pred)
    
    # Compute precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'f1': f1,
        'accuracy': accuracy
    }



def get_top_n_features(X_train, y_train,X_test, y_test, min_activity, n_features):
    y_train = y_train > min_activity
    y_test = y_test > min_activity

    # Calculate mean of high activity sequences
    high_activity_mean = X_train[y_train == 1].mean(axis=0)
    
    # Calculate mean of low activity sequences
    low_activity_mean = X_train[y_train == 0].mean(axis=0)
    
    # Calculate difference of means
    diff_of_means = high_activity_mean - low_activity_mean

    top_n = np.argsort(diff_of_means)[-n_features:]
    top_n_diff_of_means = diff_of_means[top_n]

    preds = predict_class(X_test, y_test, top_n, top_n_diff_of_means)

    metrics = compute_metrics(preds, y_test)



    return top_n, top_n_diff_of_means, metrics



def predict_class(X, y, features, thresholds):

    # Calculate mean of high activity sequences
    top_n = X[:,features]

    


    

    higher_lower = top_n > thresholds
    score = higher_lower.mean(axis=1)
    
    pred = score>0.5


    return pred





all_top_n_features = []
all_metrics = []
for min_activity in np.arange(2, 5, 0.5):
    top_n_features, top_n_diff_of_means, metrics = get_top_n_features(train_features_arr, train_activity_arr, test_features_arr, test_activity_arr, min_activity, 10)
    print(metrics)

    all_top_n_features.append((top_n_features, top_n_diff_of_means))
    all_metrics.append(metrics)











    


# %%

unique_features = np.unique([feature for features, _ in all_top_n_features for feature in features])

heatmap_data = np.zeros((len(unique_features), len(all_top_n_features)))

for i, feature in enumerate(unique_features):
    for idx, (features, diff_of_means) in enumerate(all_top_n_features):
        if feature in features:
            heatmap_data[i, idx] = diff_of_means[np.where(features == feature)[0][0]]

# %%

df = pd.DataFrame(heatmap_data, index=unique_features, columns=np.arange(1, 10, 0.5))

plt.figure(figsize=(12, 8))
plt.title("Feature Presence Across Activity Thresholds")
plt.xlabel("Activity Threshold")
plt.ylabel("Feature Index")
sns.heatmap(df, annot=False, fmt='d', cmap="Blues", mask=df == 0)
plt.show()

# %%
# For each feature display the distribution of values for high and low activity

# Create a grid of subplots
n_features = len(unique_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for idx, feature in enumerate(unique_features):
    # Plot density distributions on same axis
    sns.kdeplot(data=train_features_arr[train_activity_arr > 2, feature],
                label="High Activity", color="blue", ax=axes[idx])
    sns.kdeplot(data=train_features_arr[train_activity_arr <= 2, feature],
                label="Low Activity", color="red", ax=axes[idx])
    
    axes[idx].set_title(f"Distribution of Feature {feature}")
    axes[idx].set_xlabel("Feature Value") 
    axes[idx].set_ylabel("Density")
    axes[idx].legend()

# Remove empty subplots
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()


# %%





# %%
#  Sparse Probing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score


def sparse_logistic_regression(X_train, y_train, X_test, y_test, n_folds=5, min_activity=2):
    # Split data into high and low activity

    y_train = y_train > min_activity
    y_test = y_test > min_activity

    lr = LogisticRegressionCV(cv=5,
                                penalty="l1",
                                solver="liblinear",
                                class_weight="balanced",
                                fit_intercept=False,
                                Cs = [1e-3],
                                max_iter=10000)
    lr.fit(X_train, y_train)

    metrics = compute_metrics(lr, X_test, y_test)




    return lr, metrics

def compute_metrics(lr, X_test, y_test):
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)   
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Compute classification report
    class_report = classification_report(y_test, y_pred)
    
    # Compute precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


    
lr, metrics = sparse_logistic_regression(train_features_arr, train_activity_arr,
                                          test_features_arr, test_activity_arr)




# %%    
def baseline_zero(X, feat):
    X_base = X.copy()

    X_base[:, feat] = 0
    return X_base

def baseline_avg(X, feat):
    X_base = X.copy()
    X_base[:, feat] = X[:, feat].mean()
    return X_base

def baseline_zero_or_avg(X, y, feat):
    X_base = X.copy()
    mask_nonzero = (y != 0)
    mean_nonzero = X_base[mask_nonzero, feat].mean() if mask_nonzero.any() else 0
    X_base[mask_nonzero, feat] = mean_nonzero
    X_base[~mask_nonzero, feat] = 0
    return X_base

# --- function to compute average log-prob difference ---
def avg_log_prob_diff(X, y, model, feat, baseline_fn):
    X_orig = X
    if baseline_fn is baseline_zero_or_avg:
        X_base = baseline_fn(X, y, feat)
    else:
        X_base = baseline_fn(X, feat)

    logp_orig = np.log(model.predict_proba(X_orig)[:, 1])
    logp_base = np.log(model.predict_proba(X_base)[:, 1])
    return np.mean(logp_orig - logp_base)

coefs = lr.coef_[0]
features = np.where(coefs != 0)[0]
X_sub = train_features_arr[:1000]
y_sub = train_activity_arr[:1000]

# Compute differences
diff_zero = [avg_log_prob_diff(X_sub, y_sub, lr, feat, baseline_zero) for feat in features]
diff_avg = [avg_log_prob_diff(X_sub, y_sub, lr, feat, baseline_avg) for feat in features]
diff_zero_or_avg = [avg_log_prob_diff(X_sub, y_sub, lr, feat, baseline_zero_or_avg) for feat in features]

diff_zero = np.array(diff_zero)
diff_avg = np.array(diff_avg)
diff_zero_or_avg = np.array(diff_zero_or_avg)

# Plotting
x = np.arange(len(features))
width = 0.25

plt.figure()
plt.bar(x - width, diff_zero, width, label='Zero baseline')
plt.bar(x, diff_avg, width, label='Average baseline')
plt.bar(x + width, diff_zero_or_avg, width, label='Zero or Avg baseline')

plt.xticks(x, features)
plt.xlabel('Feature index')
plt.ylabel('Average log-prob difference')
plt.legend()
plt.title('Difference in Log-Probabilities by Feature and Baseline')
plt.tight_layout()
plt.show()











# %%
