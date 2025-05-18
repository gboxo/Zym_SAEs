import pandas as pd
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)
from collections import defaultdict

class LatentScoring:
    """
    Analyzes latent features to find correlations with sequence activity
    using ablation and steering-based methods.
    """

    def __init__(self, mean_features_path, activity_df_path, test_size=0.2, random_state=42):
        """
        Initializes the LatentScoring class by loading and preparing data.

        Args:
            mean_features_path (str): Path to the pickle file containing the mean features dictionary.
            activity_df_path (str): Path to the CSV file containing activity data.
            test_size (float): Fraction of data to use for the test set.
            random_state (int): Random seed for reproducible train/test splits.
        """
        self.mean_features_path = mean_features_path
        self.activity_df_path = activity_df_path
        self.test_size = test_size
        self.random_state = random_state

        self._load_data()
        self._preprocess_data()
        self._split_data()

        self.ablation_cv_results_df = None
        self.ablation_best_params = None
        self.final_ablation_features = None
        self.ablation_activity_threshold = None
        self.ablation_feature_indices = None
        self.optimal_pct_threshold = None
        self.ablation_test_metrics = None
        self.ablation_feature_importance_df = None

        self.diff_means_results = {} # Store results per activity threshold

        self.sparse_probe_model = None
        self.probe_activity_threshold = None
        self.sparse_probe_test_metrics = None
        self.sparse_probe_feature_importance = None


    # --- 1. Data Loading and Preparation ---

    def _load_data(self):
        """Loads features and activity data."""
        print(f"Loading mean features from: {self.mean_features_path}")
        with open(self.mean_features_path, "rb") as f:
            # Assuming features are stored directly as a dict {key: np.array}
            self.mean_features = pkl.load(f)
        print(f"Loading activity data from: {self.activity_df_path}")
        self.activity_df = pd.read_csv(self.activity_df_path)

    def _preprocess_data(self):
        """Creates activity dictionary and aligns keys."""
        # Assuming 'mutant' and 'activity_dp7' columns exist
        self.activity_dict = {
            row["mutant"]: row["activity_dp7"]
            for _, row in self.activity_df.iterrows()
            if row["mutant"] in self.mean_features
        }
        # Ensure mean_features only contains keys present in activity_dict
        self.mean_features = {
            key: feat for key, feat in self.mean_features.items() if key in self.activity_dict
        }
        self.all_keys = list(self.mean_features.keys())
        if not self.all_keys:
            raise ValueError("No common keys found between features and activity data.")
        print(f"Found {len(self.all_keys)} common keys.")

    def _split_data(self):
        """Splits data into training and testing sets."""
        if self.test_size <= 0 or self.test_size >= 1:
             # Use all data for training if test_size is invalid
            print("Warning: Invalid test_size. Using all data for training.")
            self.train_keys = self.all_keys
            self.test_keys = []
        elif len(self.all_keys) < 2:
             print("Warning: Not enough data to split. Using all data for training.")
             self.train_keys = self.all_keys
             self.test_keys = []
        else:
            self.train_keys, self.test_keys = train_test_split(
                self.all_keys,
                test_size=self.test_size,
                random_state=self.random_state
            )

        # Create feature and activity arrays
        self.train_features_arr = np.array([self.mean_features[key] for key in self.train_keys])
        self.train_activity_arr = np.array([self.activity_dict[key] for key in self.train_keys])

        if self.test_keys:
            self.test_features_arr = np.array([self.mean_features[key] for key in self.test_keys])
            self.test_activity_arr = np.array([self.activity_dict[key] for key in self.test_keys])
        else:
            self.test_features_arr = np.array([])
            self.test_activity_arr = np.array([])

        print(f"Training set size: {len(self.train_keys)}")
        print(f"Testing set size: {len(self.test_keys)}")
        if self.train_features_arr.size > 0:
             print(f"Feature array shape: {self.train_features_arr.shape}")


    # --- 2. Ablation Method ---

    def _get_top_x_percent_features(self, features_arr, activity_arr, min_activity, min_rest_fraction):
        """Internal: Finds features absent in high-activity, present in others."""
        if features_arr.shape[0] == 0: return np.array([], dtype=int) # Handle empty input

        top_mask = activity_arr >= min_activity
        rest_mask = ~top_mask

        top_features = features_arr[top_mask]
        rest_features = features_arr[rest_mask]

        # Handle cases where one group is empty
        if top_features.shape[0] == 0:
             top_zero = np.zeros(features_arr.shape[1], dtype=bool) # No top features to be zero in
        else:
             top_zero = np.all(top_features == 0, axis=0)

        if rest_features.shape[0] == 0:
             rest_sufficient = np.zeros(features_arr.shape[1], dtype=bool) # No rest features to fire
        else:
             rest_firing = np.sum(rest_features > 0, axis=0) / rest_features.shape[0]
             rest_sufficient = rest_firing >= min_rest_fraction

        return np.where(top_zero & rest_sufficient)[0]

    def _evaluate_classifier_balanced(self, predictions, ground_truth, threshold):
        """Internal: Calculates balanced evaluation metrics."""
        actual_classes = ground_truth >= threshold
        if len(predictions) == 0: # Handle empty predictions
             return {
                 "accuracy": 0, "balanced_accuracy": 0, "sensitivity": 0,
                 "specificity": 0, "precision": 0, "recall": 0, "f1": 0,
                 "true_positives": 0, "false_positives": 0, "true_negatives": 0,
                 "false_negatives": 0, "pos_count": np.sum(actual_classes),
                 "neg_count": len(actual_classes) - np.sum(actual_classes)
            }


        true_positives = np.sum((predictions == True) & (actual_classes == True))
        false_positives = np.sum((predictions == True) & (actual_classes == False))
        true_negatives = np.sum((predictions == False) & (actual_classes == False))
        false_negatives = np.sum((predictions == False) & (actual_classes == True))

        accuracy = (true_positives + true_negatives) / len(predictions) if len(predictions) > 0 else 0

        tp_fn = true_positives + false_negatives
        tn_fp = true_negatives + false_positives
        tp_fp = true_positives + false_positives
        precision_sens = 0 # Avoid division by zero later

        sensitivity = true_positives / tp_fn if tp_fn > 0 else 0
        specificity = true_negatives / tn_fp if tn_fp > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        precision = true_positives / tp_fp if tp_fp > 0 else 0

        if precision + sensitivity > 0:
            precision_sens = precision + sensitivity
            f1 = 2 * (precision * sensitivity) / precision_sens
        else:
             f1 = 0

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": sensitivity,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "pos_count": tp_fn,
            "neg_count": tn_fp
        }

    def _find_optimal_ablation_threshold(self, features_arr, activity_arr, feature_indices, activity_threshold, threshold_range=np.linspace(0.5, 1.0, 11)):
        """Internal: Finds optimal percentage threshold for feature absence."""
        if len(feature_indices) == 0 or features_arr.shape[0] == 0:
            # Return default values if no features or no data
            return 0.5, self._evaluate_classifier_balanced(np.array([]), activity_arr, activity_threshold)

        results = []
        feature_counts = np.sum(features_arr[:, feature_indices] == 0, axis=1)
        max_features = len(feature_indices)

        for percent_threshold in threshold_range:
            feature_threshold = int(max_features * percent_threshold)
            # Handle edge case: if threshold is max_features, need exact equality
            if feature_threshold == max_features and percent_threshold == 1.0:
                 predictions = feature_counts == feature_threshold
            else:
                 predictions = feature_counts >= feature_threshold

            metrics = self._evaluate_classifier_balanced(predictions, activity_arr, activity_threshold)
            metrics["percent_threshold"] = percent_threshold
            results.append(metrics)

        results_df = pd.DataFrame(results)
        if results_df.empty:
             return 0.5, self._evaluate_classifier_balanced(np.array([]), activity_arr, activity_threshold)

        best_idx = results_df["balanced_accuracy"].idxmax()
        best_row = results_df.iloc[best_idx]

        return best_row["percent_threshold"], best_row.to_dict()


    def find_ablation_features_cv(self, min_rest_fractions=[0.01, 0.05, 0.1, 0.2], activity_thresholds=np.arange(1, 7, 0.5), n_folds=5):
        """
        Performs cross-validation grid search to find optimal ablation parameters.

        Args:
            min_rest_fractions (list): List of min_rest_fraction values to test.
            activity_thresholds (list/np.array): List of min_activity values to test.
            n_folds (int): Number of cross-validation folds.

        Returns:
            pd.DataFrame: DataFrame containing the CV results for each parameter combination.
                          Returns None if no valid results found.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        grid_results = []

        for min_rest_fraction in min_rest_fractions:
            for min_activity in activity_thresholds:
                print(f"CV: Evaluating min_rest_fraction={min_rest_fraction}, min_activity={min_activity}")
                fold_results = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_features_arr)):
                    X_train_fold, X_val_fold = self.train_features_arr[train_idx], self.train_features_arr[val_idx]
                    y_train_fold, y_val_fold = self.train_activity_arr[train_idx], self.train_activity_arr[val_idx]

                    selected_features = self._get_top_x_percent_features(
                        X_train_fold, y_train_fold, min_activity, min_rest_fraction)

                    if len(selected_features) == 0: continue

                    # Find optimal pct threshold on validation data for this fold
                    opt_pct_threshold, opt_metrics = self._find_optimal_ablation_threshold(
                        X_val_fold, y_val_fold, selected_features, min_activity)

                    fold_result = {
                        "fold": fold,
                        "num_features": len(selected_features),
                        "percent_threshold": opt_pct_threshold,
                        **opt_metrics
                    }
                    fold_results.append(fold_result)

                if not fold_results: continue # Skip if no features found in any fold

                # Aggregate results across folds
                metrics_to_agg = ["balanced_accuracy", "sensitivity", "specificity", "f1", "accuracy"]
                agg_results = {
                    "min_activity": min_activity,
                    "min_rest_fraction": min_rest_fraction,
                    "num_valid_folds": len(fold_results),
                    "avg_num_features": np.mean([r["num_features"] for r in fold_results]),
                    "avg_percent_threshold": np.mean([r["percent_threshold"] for r in fold_results])
                }
                for metric in metrics_to_agg:
                    values = [r[metric] for r in fold_results]
                    agg_results[f"avg_{metric}"] = np.mean(values)
                    agg_results[f"std_{metric}"] = np.std(values)

                grid_results.append(agg_results)

        if not grid_results:
             print("No valid results found during CV.")
             self.ablation_cv_results_df = None
             self.ablation_best_params = None
             return None

        self.ablation_cv_results_df = pd.DataFrame(grid_results)
        best_idx = self.ablation_cv_results_df["avg_balanced_accuracy"].idxmax()
        self.ablation_best_params = self.ablation_cv_results_df.iloc[best_idx].to_dict()

        print("\nBest parameter combination from CV:")
        print(f"  min_activity = {self.ablation_best_params['min_activity']}")
        print(f"  min_rest_fraction = {self.ablation_best_params['min_rest_fraction']}")
        print(f"  Avg balanced accuracy: {self.ablation_best_params['avg_balanced_accuracy']:.4f}")

        return self.ablation_cv_results_df

    def select_final_ablation_features(self, min_activity=None, min_rest_fraction=None):
        """
        Selects the final set of ablation features using the best CV parameters or provided ones.

        Args:
            min_activity (float, optional): Override the best min_activity. Defaults to None.
            min_rest_fraction (float, optional): Override the best min_rest_fraction. Defaults to None.

        Returns:
            np.array: Array of selected feature indices. Returns empty array if no features found.
        """
        if min_activity is None or min_rest_fraction is None:
            if self.ablation_best_params is None:
                raise ValueError("CV must be run first or min_activity and min_rest_fraction must be provided.")
            best_min_activity = self.ablation_best_params['min_activity']
            best_min_rest_fraction = self.ablation_best_params['min_rest_fraction']
            print(f"Using best parameters from CV: min_activity={best_min_activity}, min_rest_fraction={best_min_rest_fraction}")
        else:
            best_min_activity = min_activity
            best_min_rest_fraction = min_rest_fraction
            print(f"Using provided parameters: min_activity={best_min_activity}, min_rest_fraction={best_min_rest_fraction}")


        self.final_ablation_features = self._get_top_x_percent_features(
            self.train_features_arr, self.train_activity_arr,
            best_min_activity, best_min_rest_fraction
        )
        self.ablation_activity_threshold = best_min_activity # Store the threshold used
        self.ablation_feature_indices = self.final_ablation_features # Store features

        print(f"Selected {len(self.final_ablation_features)} final ablation features.")
        #print(f"Feature indices: {self.final_ablation_features}")
        return self.final_ablation_features


    def train_ablation_classifier(self, threshold_range=np.linspace(0.5, 1.0, 21)):
         """
         Trains the ablation classifier by finding the optimal percentage threshold
         on the training data using the selected features.

         Args:
             threshold_range (np.array): Range of percentage thresholds to test.

         Returns:
             float: The optimal percentage threshold found.
                    Returns None if no features were selected previously.
         """
         if self.final_ablation_features is None or self.ablation_activity_threshold is None:
              raise ValueError("`select_final_ablation_features` must be run first.")
         if len(self.final_ablation_features) == 0:
              print("Warning: No ablation features selected. Cannot train classifier.")
              self.optimal_pct_threshold = None
              return None

         print(f"Finding optimal percentage threshold on training data for activity >= {self.ablation_activity_threshold}...")
         self.optimal_pct_threshold, train_metrics = self._find_optimal_ablation_threshold(
              self.train_features_arr, self.train_activity_arr,
              self.final_ablation_features, self.ablation_activity_threshold,
              threshold_range=threshold_range
         )

         print(f"Optimal percentage threshold found: {self.optimal_pct_threshold:.3f} "
               f"(Balanced Acc on Train: {train_metrics.get('balanced_accuracy', 0):.4f})")
         return self.optimal_pct_threshold


    def predict_ablation(self, features_arr):
        """
        Makes predictions using the trained ablation classifier.

        Args:
            features_arr (np.array): Feature array for which to make predictions.

        Returns:
            np.array: Boolean array of predictions (True = predicted high activity).
                      Returns None if the classifier hasn't been trained.
        """
        if self.optimal_pct_threshold is None or self.ablation_feature_indices is None:
            print("Warning: Ablation classifier not trained. Call `train_ablation_classifier` first.")
            return None
        if len(self.ablation_feature_indices) == 0:
             print("Warning: No features used by the ablation classifier. Returning False for all.")
             return np.zeros(features_arr.shape[0], dtype=bool)

        feature_counts = np.sum(features_arr[:, self.ablation_feature_indices] == 0, axis=1)
        max_features = len(self.ablation_feature_indices)
        feature_threshold = int(max_features * self.optimal_pct_threshold)

        # Handle edge case: if threshold is max_features, need exact equality
        if feature_threshold == max_features and self.optimal_pct_threshold == 1.0:
            predictions = feature_counts == feature_threshold
        else:
            predictions = feature_counts >= feature_threshold

        return predictions

    def evaluate_ablation_classifier(self, features_arr=None, activity_arr=None):
        """
        Evaluates the trained ablation classifier on provided data or the test set.

        Args:
            features_arr (np.array, optional): Features to evaluate on. Defaults to test set.
            activity_arr (np.array, optional): Activities to evaluate on. Defaults to test set.

        Returns:
            dict: Dictionary of balanced evaluation metrics. Returns None if evaluation fails.
        """
        if features_arr is None or activity_arr is None:
            if self.test_features_arr.size == 0 or self.test_activity_arr.size == 0:
                 print("Warning: No test data available and no data provided for evaluation.")
                 return None
            features_arr = self.test_features_arr
            activity_arr = self.test_activity_arr
            print("Evaluating ablation classifier on the test set...")
        else:
             print("Evaluating ablation classifier on provided data...")

        if self.ablation_activity_threshold is None:
             print("Warning: Ablation activity threshold not set. Cannot evaluate.")
             return None

        predictions = self.predict_ablation(features_arr)
        if predictions is None:
            return None # Prediction failed

        self.ablation_test_metrics = self._evaluate_classifier_balanced(
            predictions, activity_arr, self.ablation_activity_threshold
        )

        print("\nAblation Classifier Performance:")
        print(f"  Activity Threshold: {self.ablation_activity_threshold}")
        print(f"  Feature Indices: {self.ablation_feature_indices if self.ablation_feature_indices is not None else 'N/A'}")
        print(f"  Optimal Percentage Threshold: {self.optimal_pct_threshold if self.optimal_pct_threshold is not None else 'N/A'}")
        for key, value in self.ablation_test_metrics.items():
            if isinstance(value, (int, float)):
                 print(f"  {key.replace('_', ' ').capitalize()}: {value:.4f}")

        return self.ablation_test_metrics

    def calculate_ablation_feature_importance(self, features_arr=None, activity_arr=None):
        """
        Calculates importance of each selected ablation feature based on individual performance.

        Args:
            features_arr (np.array, optional): Features to evaluate on. Defaults to test set.
            activity_arr (np.array, optional): Activities to evaluate on. Defaults to test set.

        Returns:
            pd.DataFrame: DataFrame with importance metrics for each feature. Returns None if calculation fails.
        """
        if features_arr is None or activity_arr is None:
             if self.test_features_arr.size == 0 or self.test_activity_arr.size == 0:
                 print("Warning: No test data available and no data provided for importance calculation.")
                 return None
             features_arr = self.test_features_arr
             activity_arr = self.test_activity_arr
             print("Calculating ablation feature importance on the test set...")
        else:
              print("Calculating ablation feature importance on provided data...")


        if self.final_ablation_features is None or len(self.final_ablation_features) == 0 or self.ablation_activity_threshold is None:
            print("Warning: No final ablation features selected or activity threshold not set.")
            return None

        feature_importance = []
        for feature_idx in self.final_ablation_features:
            # Predict based on absence of this single feature
            single_feature_absence = features_arr[:, feature_idx] == 0
            metrics = self._evaluate_classifier_balanced(
                single_feature_absence, activity_arr, self.ablation_activity_threshold
            )
            feature_importance.append({
                "feature_idx": feature_idx,
                "balanced_acc": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"]
            })

        if not feature_importance:
             self.ablation_feature_importance_df = None
             return None

        self.ablation_feature_importance_df = pd.DataFrame(feature_importance)
        self.ablation_feature_importance_df = self.ablation_feature_importance_df.sort_values(
            "balanced_acc", ascending=False
        )
        print("\nTop 5 most important ablation features (individually):")
        print(self.ablation_feature_importance_df.head())
        return self.ablation_feature_importance_df

    # --- 3. Steering Methods (Difference of Means & Sparse Probes) ---

    def _compute_clf_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Internal: Computes standard classification metrics."""
        # Ensure boolean/int format
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        roc_auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None and len(np.unique(y_true)) > 1 else None
        conf_matrix = confusion_matrix(y_true, y_pred)
        try:
            # May fail if only one class predicted/present
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        except ValueError:
             class_report = "Classification report failed (likely due to single class)."
             precision, recall, f1 = 0,0,0


        accuracy = accuracy_score(y_true, y_pred)

        return {
            'roc_auc': roc_auc,
            'conf_matrix': conf_matrix,
            'class_report': class_report, # Can be dict or string
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }


    def find_steering_features_diff_means(self, min_activity, n_features=10, evaluate=True):
        """
        Finds steering features based on difference of means and evaluates.

        Args:
            min_activity (float): Activity threshold to differentiate high/low.
            n_features (int): Number of top features to select.
            evaluate (bool): Whether to evaluate the simple classifier on the test set.

        Returns:
            tuple: (top_n_indices, top_n_diff_of_means, metrics)
                   Metrics dict is None if evaluate is False or evaluation fails.
        """
        print(f"\nFinding {n_features} steering features using difference of means (Activity >= {min_activity})...")
        y_train_binary = self.train_activity_arr >= min_activity

        # Handle cases with no positive or no negative examples in training
        if np.sum(y_train_binary) == 0:
             print("Warning: No positive examples in training set for this threshold.")
             high_activity_mean = np.zeros(self.train_features_arr.shape[1])
        else:
            high_activity_mean = self.train_features_arr[y_train_binary].mean(axis=0)

        if np.sum(~y_train_binary) == 0:
             print("Warning: No negative examples in training set for this threshold.")
             low_activity_mean = np.zeros(self.train_features_arr.shape[1])
        else:
            low_activity_mean = self.train_features_arr[~y_train_binary].mean(axis=0)


        diff_of_means = high_activity_mean - low_activity_mean

        # Ensure n_features is not larger than available features
        actual_n_features = min(n_features, len(diff_of_means))
        if actual_n_features < n_features:
             print(f"Warning: Requested {n_features} features, but only {actual_n_features} are available.")

        if actual_n_features == 0:
             print("Warning: No features available to select.")
             return np.array([]), np.array([]), None


        # Get indices of top N features based on the difference
        # Use argpartition for efficiency if n_features << total features
        # If n_features is small, argsort is fine.
        if actual_n_features < len(diff_of_means) // 2:
             top_n_indices = np.argpartition(diff_of_means, -actual_n_features)[-actual_n_features:]
             # Sort the selected indices by their diff_of_means value
             top_n_indices = top_n_indices[np.argsort(diff_of_means[top_n_indices])][::-1]
        else:
            top_n_indices = np.argsort(diff_of_means)[-actual_n_features:][::-1] # Descending order

        top_n_diff_of_means = diff_of_means[top_n_indices]

        metrics = None
        if evaluate:
            if self.test_features_arr.size == 0 or self.test_activity_arr.size == 0:
                 print("Warning: No test data to evaluate difference of means classifier.")
            else:
                print("Evaluating difference of means classifier on test set...")
                y_test_binary = self.test_activity_arr >= min_activity
                test_features_selected = self.test_features_arr[:, top_n_indices]

                # Simple prediction: predict True if mean activation of selected features > 0.5 * mean diff?
                # The original code had a threshold comparison - let's refine the prediction.
                # Predict positive if the feature value is greater than the *low activity mean* for that feature?
                # Or predict positive if value > (high_mean + low_mean) / 2 ?
                # Let's try the original approach: predict positive if > 50% features are > threshold
                # What is the threshold? The diff_of_means value itself was used in the original script.
                # This seems odd, as diff_of_means is not a feature activation threshold.

                # REVISED PREDICTION (based on script logic):
                # Predict positive if > 50% of selected features have activation > the corresponding diff_of_means value.
                # This still feels heuristic. Let's stick to the script's apparent logic for now.
                if test_features_selected.shape[1] > 0: # Check if features were selected
                     higher_than_threshold = test_features_selected > top_n_diff_of_means
                     score = higher_than_threshold.mean(axis=1)
                     preds = score > 0.5 # Predict True if more than half the features exceed their diff threshold
                else:
                     preds = np.zeros(len(y_test_binary), dtype=bool) # Predict False if no features

                # Note: This prediction logic might need refinement. A simple logistic regression
                # on these selected features might be more robust.

                metrics = self._compute_clf_metrics(y_test_binary, preds)
                print("Difference of Means - Test Set Performance:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")

        # Store results keyed by activity threshold
        self.diff_means_results[min_activity] = {
            'top_n_indices': top_n_indices,
            'top_n_diff_of_means': top_n_diff_of_means,
            'metrics': metrics
        }

        return top_n_indices, top_n_diff_of_means, metrics


    def train_sparse_probe(self, min_activity, Cs=[1e-3], n_folds=5, penalty='l1', solver='liblinear', max_iter=10000, class_weight='balanced'):
        """
        Trains a sparse logistic regression probe.

        Args:
            min_activity (float): Activity threshold for binary classification.
            Cs (list/int): Regularization strengths to try in CV.
            n_folds (int): Number of folds for internal CV of LogisticRegressionCV.
            penalty (str): Regularization penalty ('l1' or 'l2').
            solver (str): Solver to use. 'liblinear' works well with L1.
            max_iter (int): Maximum iterations for solver.
            class_weight (str/dict): Class weighting strategy.

        Returns:
            tuple: (trained_model, evaluation_metrics_on_test)
                   Metrics are None if no test data exists.
        """
        print(f"\nTraining sparse probe (Logistic Regression) for activity >= {min_activity}...")
        self.probe_activity_threshold = min_activity
        y_train_binary = self.train_activity_arr >= min_activity

        # Check for sufficient data and classes
        if len(self.train_keys) < n_folds * 2 or len(np.unique(y_train_binary)) < 2 :
             print("Warning: Not enough training data or only one class present. Skipping sparse probe training.")
             self.sparse_probe_model = None
             self.sparse_probe_test_metrics = None
             return None, None


        lr = LogisticRegressionCV(
            cv=n_folds,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            fit_intercept=False, # As in original script
            Cs=[Cs] if isinstance(Cs, (int, float)) else Cs,
            max_iter=max_iter,
            random_state=self.random_state,
            scoring='roc_auc' # Use AUC for CV scoring
        )
        lr.fit(self.train_features_arr, y_train_binary)

        self.sparse_probe_model = lr
        coefs = lr.coef_[0]
        selected_features = np.where(coefs != 0)[0]
        print(f"Sparse probe trained. Selected {len(selected_features)} features with C={lr.C_[0]:.4f}")
        #print(f"Feature indices: {selected_features}")

        # Evaluate on test set
        self.sparse_probe_test_metrics = None
        if self.test_features_arr.size > 0 and self.test_activity_arr.size > 0:
             print("Evaluating sparse probe on test set...")
             y_test_binary = self.test_activity_arr >= min_activity
             if len(np.unique(y_test_binary)) < 2:
                  print("Warning: Only one class present in test set. Metrics might be limited.")
                  y_pred_proba = None # ROC AUC not meaningful
             else:
                  y_pred_proba = lr.predict_proba(self.test_features_arr)[:, 1]

             y_pred = lr.predict(self.test_features_arr)


             self.sparse_probe_test_metrics = self._compute_clf_metrics(y_test_binary, y_pred, y_pred_proba)
             print("Sparse Probe - Test Set Performance:")
             print(f"  ROC AUC: {self.sparse_probe_test_metrics.get('roc_auc', 'N/A')}") # Might be None
             print(f"  Accuracy: {self.sparse_probe_test_metrics['accuracy']:.4f}")
             print(f"  Precision: {self.sparse_probe_test_metrics['precision']:.4f}")
             print(f"  Recall: {self.sparse_probe_test_metrics['recall']:.4f}")
             print(f"  F1: {self.sparse_probe_test_metrics['f1']:.4f}")


        return self.sparse_probe_model, self.sparse_probe_test_metrics

    def calculate_sparse_probe_feature_importance(self, data_subset_size=1000, use_test_set=False):
        """
        Calculates feature importance for the sparse probe using log-prob difference.

        Args:
            data_subset_size (int): Number of data points to use for calculation (for speed).
            use_test_set (bool): If True, use test set, otherwise use training set.

        Returns:
            dict: Dictionary containing importance results ('features', 'diff_zero', 'diff_avg', 'diff_zero_or_avg').
                  Returns None if model not trained or no features selected.
        """
        if self.sparse_probe_model is None:
             print("Warning: Sparse probe model not trained.")
             return None

        coefs = self.sparse_probe_model.coef_[0]
        features = np.where(coefs != 0)[0]
        if len(features) == 0:
             print("Warning: Sparse probe selected no features.")
             return None

        if use_test_set:
             if self.test_features_arr.size == 0:
                  print("Warning: No test data available. Using training data instead.")
                  X_data = self.train_features_arr
                  y_data = self.train_activity_arr >= self.probe_activity_threshold # Use binary labels for baseline_zero_or_avg
             else:
                  X_data = self.test_features_arr
                  y_data = self.test_activity_arr >= self.probe_activity_threshold
        else:
             X_data = self.train_features_arr
             y_data = self.train_activity_arr >= self.probe_activity_threshold


        # Take a subset for speed
        if data_subset_size < X_data.shape[0]:
             indices = np.random.choice(X_data.shape[0], data_subset_size, replace=False)
             X_sub = X_data[indices]
             y_sub = y_data[indices] # y_sub should be the binary labels
        else:
             X_sub = X_data
             y_sub = y_data

        print(f"Calculating sparse probe feature importance using {X_sub.shape[0]} samples...")

        # --- Baseline Helper Functions (can be static methods or defined here) ---
        def baseline_zero(X, feat):
            X_base = X.copy()
            X_base[:, feat] = 0
            return X_base

        def baseline_avg(X, feat):
            X_base = X.copy()
            # Use mean of the subset being analyzed
            mean_val = X_sub[:, feat].mean()
            X_base[:, feat] = mean_val
            return X_base

        def baseline_zero_or_avg(X, y, feat):
            # y should be the binary labels (0/1 or False/True)
            X_base = X.copy()
            mask_nonzero_y = (y == 1) # Assuming y is binary 0/1

            # Calculate mean of feature where y is 1 in the subset
            mean_nonzero = X_sub[mask_nonzero_y[0:X_base.shape[0]], feat].mean() if mask_nonzero_y[0:X_base.shape[0]].any() else 0

            X_base[mask_nonzero_y[0:X_base.shape[0]], feat] = mean_nonzero
            X_base[~mask_nonzero_y[0:X_base.shape[0]], feat] = 0 # Set to 0 where y is 0
            return X_base

        # --- function to compute average log-prob difference ---
        def avg_log_prob_diff(X, y, model, feat, baseline_fn):
            # Ensure probabilities are not exactly 0 or 1 for log
            epsilon = 1e-15
            X_orig = X
            try:
                 if baseline_fn.__name__ == 'baseline_zero_or_avg':
                      X_base = baseline_fn(X, y, feat)
                 else:
                      X_base = baseline_fn(X, feat)

                 # Use predict_log_proba for numerical stability
                 logp_orig = model.predict_log_proba(X_orig)[:, 1] # Log-prob of class 1
                 logp_base = model.predict_log_proba(X_base)[:, 1] # Log-prob of class 1

                 # Clip probabilities slightly away from 0 and 1 if using predict_proba
                 # proba_orig = np.clip(model.predict_proba(X_orig)[:, 1], epsilon, 1 - epsilon)
                 # proba_base = np.clip(model.predict_proba(X_base)[:, 1], epsilon, 1 - epsilon)
                 # logp_orig = np.log(proba_orig)
                 # logp_base = np.log(proba_base)

                 diff = logp_orig - logp_base
                 # Handle potential NaNs/Infs if predict_log_proba still gives -inf
                 diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                 return np.mean(diff)

            except Exception as e:
                  print(f"Error calculating log-prob diff for feature {feat} with {baseline_fn.__name__}: {e}")
                  return 0.0 # Return 0 if error occurs

        # Compute differences
        diff_zero = np.array([avg_log_prob_diff(X_sub, y_sub, self.sparse_probe_model, feat, baseline_zero) for feat in features])
        diff_avg = np.array([avg_log_prob_diff(X_sub, y_sub, self.sparse_probe_model, feat, baseline_avg) for feat in features])
        diff_zero_or_avg = np.array([avg_log_prob_diff(X_sub, y_sub, self.sparse_probe_model, feat, baseline_zero_or_avg) for feat in features])


        self.sparse_probe_feature_importance = {
            'features': features,
            'diff_zero': diff_zero,
            'diff_avg': diff_avg,
            'diff_zero_or_avg': diff_zero_or_avg
        }
        print("Sparse probe feature importance calculated.")
        return self.sparse_probe_feature_importance


    # --- 4. Plotting Methods ---

    def plot_ablation_cv_results(self):
        """Plots the heatmap of balanced accuracy from the ablation CV grid search."""
        if self.ablation_cv_results_df is None or self.ablation_cv_results_df.empty:
            print("No ablation CV results to plot. Run `find_ablation_features_cv` first.")
            return

        try:
            heatmap_data = self.ablation_cv_results_df.pivot(
                index="min_rest_fraction",
                columns="min_activity",
                values="avg_balanced_accuracy"
            )

            plt.figure(figsize=(max(10, heatmap_data.shape[1]*0.8), max(6, heatmap_data.shape[0]*0.6))) # Dynamic sizing
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                        cbar_kws={'label': 'Average Balanced Accuracy'})
            plt.title('Ablation CV Grid Search: Balanced Accuracy')
            plt.xlabel('Activity Threshold (min_activity)')
            plt.ylabel('Minimum Rest Fraction')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting ablation CV results: {e}")
            print("CV Results DataFrame:")
            print(self.ablation_cv_results_df)


    def plot_ablation_confusion_matrix(self, data_label="Test Set"):
        """Plots the confusion matrix for the final ablation classifier."""
        if self.ablation_test_metrics is None:
            print("No ablation evaluation metrics found. Run `evaluate_ablation_classifier` first.")
            return

        # Extract counts from metrics
        tn = self.ablation_test_metrics.get('true_negatives', 0)
        fp = self.ablation_test_metrics.get('false_positives', 0)
        fn = self.ablation_test_metrics.get('false_negatives', 0)
        tp = self.ablation_test_metrics.get('true_positives', 0)

        conf_matrix = np.array([[tn, fp], # Predicted neg, Actual neg | Predicted neg, Actual pos
                                [fn, tp]])# Predicted pos, Actual neg | Predicted pos, Actual pos
        # Transpose to match common heatmap display: Rows=Actual, Cols=Predicted
        # No, the script had Rows=Predicted, Cols=Actual. Let's stick to that.
        # Row 0: Predicted Low (TN, FN)
        # Row 1: Predicted High (FP, TP)
        conf_matrix_display = np.array([[tn, fn], # Predicted Low (Actual Low, Actual High)
                                        [fp, tp]])# Predicted High (Actual Low, Actual High)


        plt.figure(figsize=(7, 5))
        sns.heatmap(conf_matrix_display, annot=True, fmt='d', cmap="Blues",
                    xticklabels=[f"Low Activity (<{self.ablation_activity_threshold})", f"High Activity (>= {self.ablation_activity_threshold})"],
                    yticklabels=["Predicted Low", "Predicted High"])
        plt.title(f"Ablation Classifier Confusion Matrix ({data_label})")
        plt.ylabel("Predicted Label")
        plt.xlabel("True Label")
        plt.tight_layout()
        plt.show()


    def plot_ablation_feature_importance(self):
        """Plots the importance of individual ablation features."""
        if self.ablation_feature_importance_df is None or self.ablation_feature_importance_df.empty:
            print("No ablation feature importance data. Run `calculate_ablation_feature_importance` first.")
            return

        df_plot = self.ablation_feature_importance_df.copy()
         # Sort by index if many features, otherwise keep sorted by importance
        if len(df_plot) > 30:
             df_plot = df_plot.sort_values("feature_idx")


        plt.figure(figsize=(max(12, len(df_plot) * 0.3), 6)) # Dynamic width
        plt.bar(df_plot["feature_idx"].astype(str), df_plot["balanced_acc"])
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label="Random Guess (0.5)")
        plt.xlabel("Feature Index")
        plt.ylabel("Balanced Accuracy (Single Feature)")
        plt.title("Ablation Method: Individual Feature Importance")
        plt.xticks(rotation=90, fontsize=8 if len(df_plot) > 30 else 10)
        plt.ylim(bottom=min(0.4, df_plot["balanced_acc"].min() - 0.02)) # Adjust y-axis floor
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_sparse_probe_confusion_matrix(self, data_label="Test Set"):
         """Plots the confusion matrix for the sparse probe."""
         if self.sparse_probe_test_metrics is None:
             print("No sparse probe evaluation metrics found. Run `train_sparse_probe` (with test data) first.")
             return
         if self.probe_activity_threshold is None:
             print("Sparse probe activity threshold not set.")
             return

         # Check if conf_matrix is valid
         conf_matrix = self.sparse_probe_test_metrics.get('conf_matrix')
         if conf_matrix is None or not isinstance(conf_matrix, np.ndarray) or conf_matrix.shape != (2,2):
              print("Sparse probe confusion matrix not available or invalid in metrics.")
              return

         # Confusion matrix from sklearn is [[TN, FP], [FN, TP]]
         # For display (Rows=Predicted, Cols=Actual):
         tn, fp, fn, tp = conf_matrix.ravel()
         conf_matrix_display = np.array([[tn, fn], # Predicted 0 (Actual 0, Actual 1)
                                         [fp, tp]])# Predicted 1 (Actual 0, Actual 1)


         plt.figure(figsize=(7, 5))
         sns.heatmap(conf_matrix_display, annot=True, fmt='d', cmap="Blues",
                    xticklabels=[f"Low Activity (<{self.probe_activity_threshold})", f"High Activity (>= {self.probe_activity_threshold})"],
                    yticklabels=["Predicted Low", "Predicted High"])
         plt.title(f"Sparse Probe Confusion Matrix ({data_label})")
         plt.ylabel("Predicted Label")
         plt.xlabel("True Label")
         plt.tight_layout()
         plt.show()


    def plot_sparse_probe_feature_importance(self):
        """Plots the feature importance for the sparse probe using log-prob differences."""
        if self.sparse_probe_feature_importance is None:
            print("No sparse probe feature importance data. Run `calculate_sparse_probe_feature_importance` first.")
            return

        data = self.sparse_probe_feature_importance
        features = data['features']
        diff_zero = data['diff_zero']
        diff_avg = data['diff_avg']
        diff_zero_or_avg = data['diff_zero_or_avg']

        if len(features) == 0:
            print("Sparse probe selected no features to plot importance for.")
            return

        x = np.arange(len(features))
        width = 0.25

        plt.figure(figsize=(max(12, len(features) * 0.5), 6)) # Dynamic width
        plt.bar(x - width, diff_zero, width, label='Zero Baseline')
        plt.bar(x, diff_avg, width, label='Average Baseline')
        plt.bar(x + width, diff_zero_or_avg, width, label='Zero or Avg Baseline')

        plt.xticks(x, features.astype(str), rotation=90, fontsize=8 if len(features) > 30 else 10)
        plt.xlabel('Feature Index')
        plt.ylabel('Average Log-Prob Difference')
        plt.legend()
        plt.title('Sparse Probe: Feature Importance (Log-Prob Difference)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_feature_distributions(self, feature_indices, activity_threshold, use_test_set=False):
        """
        Plots the distribution of specified feature values for high vs. low activity.

        Args:
            feature_indices (list/np.array): Indices of features to plot.
            activity_threshold (float): Threshold to define high/low activity.
            use_test_set (bool): If True, use test set, otherwise use training set.
        """
        if use_test_set:
             if self.test_features_arr.size == 0 or self.test_activity_arr.size == 0:
                  print("Warning: No test data available.")
                  return
             features_arr = self.test_features_arr
             activity_arr = self.test_activity_arr
             data_label = "Test Set"
        else:
             features_arr = self.train_features_arr
             activity_arr = self.train_activity_arr
             data_label = "Train Set"

        if features_arr.size == 0:
             print(f"Warning: No data in {data_label} to plot distributions.")
             return


        high_mask = activity_arr >= activity_threshold
        low_mask = ~high_mask

        n_features = len(feature_indices)
        if n_features == 0:
            print("No feature indices provided.")
            return

        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes = axes.flatten()

        for i, feat_idx in enumerate(feature_indices):
            if feat_idx >= features_arr.shape[1]:
                 print(f"Warning: Feature index {feat_idx} out of bounds. Skipping.")
                 axes[i].set_title(f"Feature {feat_idx} (Invalid)")
                 axes[i].axis('off')
                 continue

            ax = axes[i]
            feature_data = features_arr[:, feat_idx]

            # Plot KDEs only if there are samples in both classes
            if np.sum(high_mask) > 1: # Need >1 sample for KDE
                sns.kdeplot(data=feature_data[high_mask], label=f"High (>= {activity_threshold})", color="blue", ax=ax, warn_singular=False)
            else:
                 print(f"Warning: Not enough high activity samples for feature {feat_idx} KDE.")


            if np.sum(low_mask) > 1: # Need >1 sample for KDE
                sns.kdeplot(data=feature_data[low_mask], label=f"Low (< {activity_threshold})", color="red", ax=ax, warn_singular=False)
            else:
                 print(f"Warning: Not enough low activity samples for feature {feat_idx} KDE.")


            ax.set_title(f"Feature {feat_idx} Distribution ({data_label})")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Density")
            if np.sum(high_mask) > 1 or np.sum(low_mask) > 1:
                ax.legend()
            ax.grid(True, alpha=0.2)


        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


# --- Example Usage (put outside class definition) ---
if __name__ == "__main__":
    # Adjust paths as needed
    feature_path = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/latent_scoring/latent_scoring_0_new/features/features_M0_D0.pkl"
    activity_path = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/alpha-amylase-training-data.csv"

    with open(feature_path, "rb") as f:
        features = pkl.load(f)
    
    features = {key: np.array(features[key].todense()).sum(axis=0) for key in features.keys()}

    with open("/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/latent_scoring/latent_scoring_0_new/features/mean_features_M0_D0.pkl", "wb") as f:
        pkl.dump(features, f)




    # Check if files exist
    if not os.path.exists(feature_path):
        print(f"Error: Feature file not found at {feature_path}")
    elif not os.path.exists(activity_path):
        print(f"Error: Activity file not found at {activity_path}")
    else:
        print("Starting LatentScoring Analysis...")
        scorer = LatentScoring(mean_features_path=feature_path, activity_df_path=activity_path, random_state=42)

        # --- Ablation Method Example ---
        print("\n--- Running Ablation Method ---")
        # 1. Run Cross-Validation Grid Search
        cv_results = scorer.find_ablation_features_cv(
             min_rest_fractions=[0.05, 0.1, 0.2], # Reduced grid for example
             activity_thresholds=np.arange(2, 6, 1.0) # Reduced grid for example
        )
        if cv_results is not None:
            scorer.plot_ablation_cv_results()

            # 2. Select final features using best params from CV
            selected_ablation_features = scorer.select_final_ablation_features()

            if selected_ablation_features is not None and len(selected_ablation_features) > 0:
                # 3. Train classifier (find optimal percentage threshold)
                scorer.train_ablation_classifier()

                # 4. Evaluate on test set
                scorer.evaluate_ablation_classifier()
                scorer.plot_ablation_confusion_matrix()

                # 5. Calculate and plot feature importance
                scorer.calculate_ablation_feature_importance()
                scorer.plot_ablation_feature_importance()

                # 6. Plot distributions for selected features
                scorer.plot_feature_distributions(
                     feature_indices=selected_ablation_features[:min(len(selected_ablation_features), 8)], # Plot first few
                     activity_threshold=scorer.ablation_activity_threshold
                 )
            else:
                 print("No ablation features selected based on CV results.")


        # --- Steering Method Example (Sparse Probe) ---
        print("\n--- Running Steering Method (Sparse Probe) ---")
        probe_activity_threshold = 4.0 # Example threshold
        # 1. Train sparse probe
        model, metrics = scorer.train_sparse_probe(min_activity=probe_activity_threshold, Cs=[1e-4, 1e-3, 1e-2]) # Example Cs

        if model is not None and metrics is not None:
             # 2. Plot confusion matrix (if evaluated)
             scorer.plot_sparse_probe_confusion_matrix()

             # 3. Calculate and plot feature importance
             scorer.calculate_sparse_probe_feature_importance()
             scorer.plot_sparse_probe_feature_importance()

             # 4. Plot distributions for selected features (if any)
             probe_features = scorer.sparse_probe_feature_importance['features'] if scorer.sparse_probe_feature_importance else []
             if len(probe_features) > 0:
                  scorer.plot_feature_distributions(
                        feature_indices=probe_features[:min(len(probe_features), 8)], # Plot first few
                        activity_threshold=probe_activity_threshold
                  )
        else:
             print(f"Sparse probe training/evaluation failed for threshold {probe_activity_threshold}.")


        # --- Steering Method Example (Difference of Means) ---
        print("\n--- Running Steering Method (Difference of Means) ---")
        diff_means_activity_threshold = 4.0 # Example threshold
        # 1. Find features and evaluate
        indices, diffs, diff_metrics = scorer.find_steering_features_diff_means(
             min_activity=diff_means_activity_threshold,
             n_features=15,
             evaluate=True
         )
        if indices is not None and len(indices) > 0:
             # 2. Plot distributions for selected features
             scorer.plot_feature_distributions(
                 feature_indices=indices[:min(len(indices), 8)], # Plot first few
                 activity_threshold=diff_means_activity_threshold
             )


        print("\nLatentScoring Analysis Complete.")
    


