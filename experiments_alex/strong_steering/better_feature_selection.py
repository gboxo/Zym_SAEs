from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import pandas as pd

import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold




class LatentScoring:
    def __init__(self, mean_features, activity_dict):

        self.mean_features = mean_features
        self.activity_dict = activity_dict
    def split_data(self, test_size=0.2):
        random_indices = np.random.permutation(len(self.mean_features))
        train_indices = random_indices[:int(len(random_indices)*0.8)]
        test_indices = random_indices[int(len(random_indices)*0.8):]

        all_keys = list(self.mean_features.keys())
        train_keys = [all_keys[i] for i in train_indices]
        test_keys = [all_keys[i] for i in test_indices]

        train_features = {key:self.mean_features[key] for key in train_keys}
        test_features = {key:self.mean_features[key] for key in test_keys}

        train_activity = {key:self.activity_dict[key] for key in train_keys}
        test_activity = {key:self.activity_dict[key] for key in test_keys}



        self.train_features_arr = np.array([train_features[key] for key in train_keys])
        self.test_features_arr = np.array([test_features[key] for key in test_keys])
        self.train_activity_arr = np.array([train_activity[key] for key in train_keys])[:,0]
        self.test_activity_arr = np.array([test_activity[key] for key in test_keys])[:,0]

    



class LatentScoringAblation(LatentScoring):
    def __init__(self, mean_features, activity_dict):
        super().__init__(mean_features, activity_dict)

    def get_features_greater_than_min_activity(self,X_train, y_train, min_activity=2, min_rest_fraction=0.01):

        """
        Get the features that don't fire for sequences with activty higher than min_activity,
        but do fire on at least min_rest_fraction of the remaining sequences.
        
        Args:
            min_activity: Minimum activity value for sequences to be considered
            min_rest_fraction: Minimum fraction of remaining sequences that must have the feature (default 0.1)
        
        Returns:
            Array of indices for features that meet the criteria
        """
        # Split into top x% and rest
        top_mask = y_train >= min_activity
        rest_mask = ~top_mask
        

        # Get features for each group
        top_features = X_train[top_mask]
        rest_features = X_train[rest_mask]
        
        # Find features that don't fire in top x%
        top_zero = np.all(top_features == 0, axis=0)
        
        # Find features that fire in at least min_rest_fraction of rest
        rest_firing = np.sum(rest_features > 0, axis=0) / rest_features.shape[0]
        rest_sufficient = rest_firing >= min_rest_fraction
        
        # Return indices where both conditions are met
        return np.where(top_zero & rest_sufficient)[0]

    def predict_activity_by_feature_absence(self, feature_indices, activity_threshold):
        """
        Predict whether activity is above/below threshold by checking absence of specific features.
        
        Args:
            activity_threshold: Threshold to classify activity as high/low
        
        Returns:
            predictions: Binary array (True = high activity, False = low activity)
        """
        # Check if the selected features are absent (= 0) in test sequences
        # If any of the selected features are absent, predict high activity; otherwise, predict low activity
        feature_absence = np.any(self.test_features_arr[:, feature_indices] == 0, axis=1)
        return feature_absence  # True means predicted high activity (above threshold)


    def evaluate_classifier_balanced(self, predictions, ground_truth, threshold):
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
    # %%
    # Implement cross-validation for feature selection and evaluation
    def cross_validate_feature_selection(self, min_activity, min_rest_fraction, n_folds=5):
        """
        Perform cross-validation for feature selection and evaluation
        
        Args:
            min_activity: Minimum activity threshold
            min_rest_fraction: Minimum fraction of remaining sequences with feature
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with cross-validation results
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []
        
        # For each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.train_features_arr)):
            # Split data into train and test
            X_train, X_test = self.train_features_arr[train_idx], self.train_features_arr[test_idx]
            y_train, y_test = self.train_activity_arr[train_idx], self.train_activity_arr[test_idx]
            
            # Select features using training data only
            selected_features = self.get_features_greater_than_min_activity(
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
                
                metrics = self.evaluate_classifier_balanced(predictions, y_test, min_activity)
                
                if metrics["balanced_accuracy"] > best_balanced_acc:
                    best_balanced_acc = metrics["balanced_accuracy"]
                    best_pct_threshold = pct_threshold
                    best_metrics = metrics
            
            # Add fold results
            if best_metrics is not None:
                fold_result = {
                    "selected_features":selected_features,
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
            "selected_features":[r["selected_features"] for r in cv_results],
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

"""
PLOT THE FEATURES PRESENCE AT DIFFERENT ACTIVITY THRESHOLDS


all_features = []
for min_activity in np.arange(1, 10, 0.1):
    features = get_top_x_percent_features(train_features_arr, train_activity_arr, min_activity = min_activity, min_rest_fraction=0.1)
    all_features.append(features)
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
"""








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


class LatentScoringSteering(LatentScoring):
    def __init__(self, mean_features, activity_dict):
        super().__init__(mean_features, activity_dict)


    def predict_class(self,features, thresholds):

        # Calculate mean of high activity sequences
        top_n = self.test_features_arr[:,features]
        higher_lower = top_n > thresholds
        score = higher_lower.mean(axis=1)
        pred = score>0.5
        return pred


    def get_top_n_features(self, min_activity, n_features):
        y_train = self.train_activity_arr > min_activity
        y_test = self.test_activity_arr > min_activity

        # Calculate mean of high activity sequences
        high_activity_mean = self.train_features_arr[y_train == 1].mean(axis=0)
        
        # Calculate mean of low activity sequences
        low_activity_mean = self.train_features_arr[y_train == 0].mean(axis=0)
        
        # Calculate difference of means
        diff_of_means = high_activity_mean - low_activity_mean

        top_n = np.argsort(diff_of_means)[-n_features:]
        top_n_diff_of_means = diff_of_means[top_n]

        preds = self.predict_class(top_n, top_n_diff_of_means)

        metrics = self.compute_metrics(preds, y_test)

        return top_n, top_n_diff_of_means, metrics
    def compute_metrics(self, preds, y_test):
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

    def sparse_logistic_regression(self, n_folds=5, min_activity=2):
        # Split data into high and low activity
        X_train = self.train_features_arr
        y_train = self.train_activity_arr > min_activity
        X_test = self.test_features_arr
        y_test = self.test_activity_arr > min_activity

        # Define the Logistic Regression model with Cross-Validation
        lr = LogisticRegressionCV(cv=n_folds, # Use n_folds argument
                                    penalty="l1",
                                    solver="liblinear",
                                    class_weight="balanced",
                                    max_iter=10000,
                                    Cs=[4e-4 ], # Example C value, can be a list
                                    )
        lr.fit(X_train, y_train)


        # Pass X_test and y_test to the metrics computation
        metrics = self.compute_metrics_model(lr, X_test, y_test)

        return lr, metrics

    # Modified to accept X_test and y_test
    def compute_metrics_model(self, lr, X_test, y_test):
        # Predictions using the passed test set
        y_pred = lr.predict(X_test)
        y_pred_proba = lr.predict_proba(X_test)[:, 1]

        # Compute ROC AUC score using the correct y_test
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Compute confusion matrix using the correct y_test
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Compute classification report using the correct y_test
        class_report = classification_report(y_test, y_pred, output_dict=True) # Get report as dict

        # Compute precision, recall, F1 score using the correct y_test
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        # Compute accuracy using the correct y_test
        accuracy = accuracy_score(y_test, y_pred)

        # Return metrics including the report dictionary
        return {
            'roc_auc': roc_auc,
            'conf_matrix': conf_matrix,
            'class_report_dict': class_report, # Store the dictionary version
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }




    def baseline_zero(self, feat, X_features):
        X_base = X_features.copy()
        X_base[:, feat] = 0
        return X_base

    def baseline_avg(self, feat, X_features):
        X_base = X_features.copy()
        X_base[:, feat] = X_base[:, feat].mean() # Use mean from the provided array
        return X_base

    def baseline_zero_or_avg(self, feat, X_features):
        # This baseline needs activity array. Assume it corresponds to X_features.
        # If X_features is test set, use self.test_activity_arr. If train, use self.train_activity_arr
        # For simplicity, let's assume it always operates on test set here, matching avg_log_prob_diff default
        if X_features is self.test_features_arr:
             activity_arr = self.test_activity_arr
        # elif X_features is self.train_features_arr: # Add if needed
        #      activity_arr = self.train_activity_arr
        else:
             # If X_features is some other subset, we can't easily get the corresponding activity
             # Defaulting to test activity, but this might need adjustment depending on usage
             print("Warning: baseline_zero_or_avg assuming test activities for provided features.")
             activity_arr = self.test_activity_arr


        X_base = X_features.copy()
        mask_nonzero = (activity_arr != 0)
        # Calculate mean only from the non-zero activity samples within the provided X_features
        mean_nonzero = X_base[mask_nonzero, feat].mean() if mask_nonzero.any() else 0
        X_base[mask_nonzero, feat] = mean_nonzero
        X_base[~mask_nonzero, feat] = 0
        return X_base

    # --- function to compute average log-prob difference ---
    # Pass X_orig explicitly if needed, otherwise default to self.test_features_arr
    def avg_log_prob_diff(self, feat, baseline_fn, lr, X_orig=None):
        if X_orig is None:
            X_orig = self.test_features_arr

        # Ensure baseline function gets the correct features array if it differs from self.test_features_arr
        # Modify baseline functions if they need X_orig passed explicitly
        X_base = baseline_fn(feat, X_orig) # Pass X_orig to baseline_fn

        # Handle potential log(0)
        eps = 1e-12
        logp_orig = np.log(lr.predict_proba(X_orig)[:, 1] + eps)
        logp_base = np.log(lr.predict_proba(X_base)[:, 1] + eps)
        return np.mean(logp_orig - logp_base)





if __name__ == "__main__":
    DMS = False

    # --- Data Loading ---
    path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base/features/"
    plotting_path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base/plots/"
    features_path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base/saved_features/" # New path for saved features
    os.makedirs(plotting_path, exist_ok=True) # Ensure the plotting directory exists
    os.makedirs(features_path, exist_ok=True) # Ensure the features directory exists

    with open(os.path.join(path, "features_M0_D0.pkl"), "rb") as f:
        features = pkl.load(f)
    mean_features = {key:np.array(val.todense()).sum(axis=0) for key,val in features.items()}
    with open(os.path.join(path, "mean_features_M0_D0.pkl"), "wb") as f:
        pkl.dump(mean_features, f)





    # Load mean features (assuming this file exists and is preferred over calculating from raw features)
    with open(os.path.join(path, "mean_features_M0_D0.pkl"), "rb") as f:
        mean_features = pkl.load(f)
    

    if DMS:
        # Load activity data
        df_path = os.path.join("/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/alpha-amylase-training-data.csv")
        df = pd.read_csv(df_path)
        activity_dict = {key:df[df["mutant"] == key]["activity_dp7"].values[0] for key in mean_features.keys()}
        activity_thresholds_ablation = np.arange(2, 6, 0.5) # Example smaller grid
        activity_thresholds_steering = np.arange(2, 5, 0.5) # Example range
    else:
        df = pd.read_csv("/home/woody/b114cb/b114cb23/boxo/strong_steering/seq_gen/activity_predictions.csv")
        df["average_activity"] = df.loc[:,["prediction1","prediction2"]].mean(axis=1)
        activity_dict = {key:df[df["index"] == key]["average_activity"].values for key in mean_features.keys()}
        activity_thresholds_ablation = np.arange(1, 2, 0.25) # Example smaller grid
        activity_thresholds_steering = np.arange(1, 2, 0.25) # Example range


    ablation_scorer = LatentScoringAblation(mean_features, activity_dict)
    steering_scorer = LatentScoringSteering(mean_features, activity_dict)

    ablation_scorer.split_data()
    steering_scorer.split_data()

    print("--- Running Ablation Analysis ---")
    if True:
        # Grid search parameters
        min_rest_fractions_ablation = [0.01,0.05, 0.1, 0.3,0.5] # Example smaller grid
        n_folds_ablation = 5

        all_cv_results_ablation = []
        print(f"Running cross-validation for Ablation Scorer over {len(activity_thresholds_ablation)} activity thresholds and {len(min_rest_fractions_ablation)} rest fractions...")
        for min_act in activity_thresholds_ablation:
            for min_rest in min_rest_fractions_ablation:
                results = ablation_scorer.cross_validate_feature_selection(
                    min_activity=min_act,
                    min_rest_fraction=min_rest,
                    n_folds=n_folds_ablation
                )
                if results["valid"]:
                    all_cv_results_ablation.append(results)

        print(f"Completed Ablation cross-validation. Found {len(all_cv_results_ablation)} valid parameter combinations.")

        if all_cv_results_ablation:
            results_df_ablation = pd.DataFrame(all_cv_results_ablation)
            print("\nAblation Cross-Validation Results Summary:")
            print(results_df_ablation.sort_values(by="avg_balanced_accuracy", ascending=False)[[
                "min_activity", "min_rest_fraction", "avg_num_features",
                "avg_balanced_accuracy", "std_balanced_accuracy",
                "avg_sensitivity", "avg_specificity"
            ]].head())

            # Save the results DataFrame
            ablation_results_filename = os.path.join(features_path, "ablation_cv_results_summary.pkl")
            with open(ablation_results_filename, "wb") as f:
                pkl.dump(results_df_ablation, f)
            print(f"Saved ablation results summary to {ablation_results_filename}")

            # Example Plotting: How avg balanced accuracy changes with parameters
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=results_df_ablation, x="min_activity", y="avg_balanced_accuracy", hue="min_rest_fraction", marker='o')
            plt.title("Ablation: Avg Balanced Accuracy vs. Activity Threshold (by Min Rest Fraction)")
            plt.xlabel("Minimum Activity Threshold")
            plt.ylabel("Average Balanced Accuracy (CV)")
            plt.legend(title="Min Rest Fraction")
            plt.grid(True)
            plt.savefig(os.path.join(plotting_path, "ablation_balanced_accuracy_vs_threshold.png"), bbox_inches='tight')
            plt.close() # Close the figure


        else:
            print("No valid results found from Ablation cross-validation.")
        print("--- Finished Ablation Analysis ---\n")


    # --- Steering Section ---
    print("--- Running Steering Analysis ---")

    # %% Difference of Means Analysis
    all_top_n_features_steering = []
    all_metrics_steering = []

    for min_activity in activity_thresholds_steering:
        # Use the method from the steering_scorer instance
        top_n, top_n_diff_of_means, metrics = steering_scorer.get_top_n_features(
            min_activity=min_activity,
            n_features=10 # Example: find top 10 features
        )
        print(f"Steering DiffMeans: min_activity={min_activity}, Accuracy={metrics['accuracy']:.4f}") # Example: print accuracy

        all_top_n_features_steering.append((top_n, top_n_diff_of_means))
        all_metrics_steering.append(metrics)

        # Save the identified features and their diff-of-means scores for this threshold
        diffmeans_features_data = {'features': top_n, 'diff_of_means': top_n_diff_of_means}
        diffmeans_filename = os.path.join(features_path, f"steering_diffmeans_features_min_activity_{min_activity:.1f}.pkl")
        with open(diffmeans_filename, "wb") as f:
            pkl.dump(diffmeans_features_data, f)
        # print(f"Saved DiffMeans features for min_activity={min_activity} to {diffmeans_filename}")


    # Feature presence heatmap (using results from get_top_n_features)
    unique_features_steering = np.unique([feature for features, _ in all_top_n_features_steering for feature in features])
    if len(unique_features_steering) > 0: # Check if any features were found
        heatmap_data_steering = np.zeros((len(unique_features_steering), len(all_top_n_features_steering)))

        for i, feature in enumerate(unique_features_steering):
            for idx, (features, diff_of_means) in enumerate(all_top_n_features_steering):
                if feature in features:
                    # Find the index of the feature within the 'features' array for this threshold
                    feature_index_in_list = np.where(features == feature)[0]
                    if len(feature_index_in_list) > 0:
                        heatmap_data_steering[i, idx] = diff_of_means[feature_index_in_list[0]]


        df_steering = pd.DataFrame(heatmap_data_steering, index=unique_features_steering, columns=activity_thresholds_steering)

        plt.figure(figsize=(12, 8))
        plt.title("Steering Feature Diff-of-Means Across Activity Thresholds")
        plt.xlabel("Activity Threshold")
        plt.ylabel("Feature Index")
        sns.heatmap(df_steering, annot=False, fmt='.2f', cmap="Blues", mask=df_steering == 0) # Use fmt '.2f' for floats
        plt.savefig(os.path.join(plotting_path, "steering_diff_means_heatmap.png"), bbox_inches='tight')
        plt.close() # Close the figure

    # %% Distribution Plots for Steering Features
    if len(unique_features_steering) > 0:
        n_features = len(unique_features_steering)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()

        example_min_activity_for_plot = activity_thresholds_steering[-1] # Choose a threshold for plotting distributions

        for idx, feature in enumerate(unique_features_steering):
            # Access data via the instance
            high_mask = steering_scorer.train_activity_arr > example_min_activity_for_plot
            low_mask = steering_scorer.train_activity_arr <= example_min_activity_for_plot

            sns.kdeplot(data=steering_scorer.train_features_arr[high_mask, feature],
                        label="High Activity", color="blue", ax=axes[idx], warn_singular=False)
            sns.kdeplot(data=steering_scorer.train_features_arr[low_mask, feature],
                        label="Low Activity", color="red", ax=axes[idx], warn_singular=False)

            axes[idx].set_title(f"Distribution of Feature {feature} (Activity > {example_min_activity_for_plot})")
            axes[idx].set_xlabel("Feature Value")
            axes[idx].set_ylabel("Density")
            axes[idx].legend()

        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plot_filename = f"steering_feature_distributions_gt_{example_min_activity_for_plot}.png"
        plt.savefig(os.path.join(plotting_path, plot_filename), bbox_inches='tight')
        plt.close() # Close the figure

    # %% Sparse Logistic Regression
    all_lr_results = [] # Store results for potential later analysis/saving
    for min_activity in activity_thresholds_steering:
        print(f"\nRunning Sparse Logistic Regression for Steering (min_activity={min_activity:.1f})...")
        # Use the method from the steering_scorer instance
        lr_model, lr_metrics = steering_scorer.sparse_logistic_regression(n_folds=5, min_activity=min_activity)
        print(f"Sparse LR Metrics (Steering, min_activity={min_activity:.1f}):")
        # Print metrics in a more readable format
        for key, value in lr_metrics.items():
            if key == 'conf_matrix':
                print(f"  {key}:\n{value}")
            elif key == 'class_report_dict':
                # Print key metrics from the report dict
                print(f"  precision (pos): {value['True']['precision']:.4f}")
                print(f"  recall (pos):    {value['True']['recall']:.4f}")
                print(f"  f1-score (pos):  {value['True']['f1-score']:.4f}")
            elif isinstance(value, float):
                 print(f"  {key}: {value:.4f}")
        # Optionally save the model itself
        model_filename = os.path.join(features_path, f"steering_sparselr_model_min_activity_{min_activity:.1f}.pkl")
        with open(model_filename, "wb") as f:
            pkl.dump(lr_model, f)

        # %% Log-Prob Difference Analysis & Feature Saving
        coefs = lr_model.coef_[0]
        selected_features_lr = np.where(coefs != 0)[0] # Renamed to avoid clash if ablation features were stored

        # Save the selected features from sparse LR
        sparselr_features_data = {'features': selected_features_lr, 'coefficients': coefs[selected_features_lr]}
        sparselr_filename = os.path.join(features_path, f"steering_sparselr_features_min_activity_{min_activity:.1f}.pkl")
        with open(sparselr_filename, "wb") as f:
            pkl.dump(sparselr_features_data, f)
        # print(f"Saved SparseLR features for min_activity={min_activity} to {sparselr_filename}")


        if len(selected_features_lr) > 0:
            print(f"\nCalculating log-prob differences for {len(selected_features_lr)} features (from Sparse LR)...")
            # Use methods from the steering_scorer instance
            # Pass X_orig=steering_scorer.test_features_arr explicitly if needed, but default should work
            diff_zero = [steering_scorer.avg_log_prob_diff(feat, steering_scorer.baseline_zero, lr_model) for feat in selected_features_lr]
            diff_avg = [steering_scorer.avg_log_prob_diff(feat, steering_scorer.baseline_avg, lr_model) for feat in selected_features_lr]
            diff_zero_or_avg = [steering_scorer.avg_log_prob_diff(feat, steering_scorer.baseline_zero_or_avg, lr_model) for feat in selected_features_lr]

            diff_zero = np.array(diff_zero)
            diff_avg = np.array(diff_avg)
            diff_zero_or_avg = np.array(diff_zero_or_avg)

            # Plotting
            x = np.arange(len(selected_features_lr))
            width = 0.25

            plt.figure(figsize=(max(10, len(selected_features_lr) * 0.5), 6)) # Adjust figure size
            plt.bar(x - width, diff_zero, width, label='Zero baseline')
            plt.bar(x, diff_avg, width, label='Average baseline')
            plt.bar(x + width, diff_zero_or_avg, width, label='Zero or Avg baseline')

            plt.xticks(x, selected_features_lr, rotation=90) # Rotate labels if many features
            plt.xlabel('Feature index')
            plt.ylabel('Average log-prob difference')
            plt.legend()
            plt.title(f'Log-Prob Diff (Sparse LR Feats, min_act={min_activity:.1f}) by Baseline') # Updated title
            plt.tight_layout()
            plt.savefig(os.path.join(plotting_path, f"steering_log_prob_difference_lr_features_min_activity_{min_activity:.1f}.png"), bbox_inches='tight')
            plt.close() # Close the figure
        else:
            print(f"\nNo features selected by sparse logistic regression for min_activity={min_activity:.1f}. Skipping log-prob diff plot.")

    print("--- Finished Steering Analysis ---")
