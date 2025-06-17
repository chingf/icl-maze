import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import pandas as pd
import configs
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.utils import find_ckpt_file, convert_to_tensor
import h5py
import random
from src.envs.darkroom import DarkroomEnv

import warnings

from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_and_evaluate_regression(X_train, Y_train, X_test, Y_test, print_scores=True, make_plot=True):
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    
    X_train_np = [np.array([_x for _x in x]) for x in X_train]
    X_test_np = [np.array([_x for _x in x]) for x in X_test]
    Y_train_np = np.array(Y_train)
    Y_test_np = np.array(Y_test)

    alphas = np.logspace(0, 4, 10)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluate_fold(X, y, train_idx, val_idx, alpha):
        # Train on this fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        pipeline.fit(X[train_idx], y[train_idx])
        # Get validation score
        val_score = pipeline.score(X[val_idx], y[val_idx])
        return val_score

    pipelines = []
    test_scores = []
    test_y = []
    test_pred = []
    
    for layer in range(len(X_train)):
        # Parallel CV for each alpha
        cv_scores = {alpha: [] for alpha in alphas}
        for alpha in alphas:
            scores = Parallel(n_jobs=-1)(
                delayed(evaluate_fold)(
                    X_train_np[layer], Y_train_np, 
                    train_idx, val_idx, alpha
                )
                for train_idx, val_idx in kf.split(X_train_np[layer])
            )
            cv_scores[alpha] = np.mean(scores)
        
        # Find best alpha
        best_alpha = max(cv_scores.items(), key=lambda x: x[1])[0]
        
        # Train final model with best alpha
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=best_alpha))
        ])
        pipeline.fit(X_train_np[layer], Y_train_np)
        
        train_score = pipeline.score(X_train_np[layer], Y_train_np)
        test_score = pipeline.score(X_test_np[layer], Y_test_np)
        pred = pipeline.predict(X_test_np[layer])
        
        pipelines.append(pipeline)
        test_scores.append(np.abs(pred - Y_test_np))
        test_y.append(Y_test_np)
        test_pred.append(pred)

        if print_scores:
            print(f"Layer {layer}:")
            print(f"Best alpha: {best_alpha:.3f}")
            print(f"Train R2: {train_score:.3f}")
            print(f"Test R2: {test_score:.3f}")
            print()

        if make_plot:
            y_pred = pipeline.predict(X_test_np[layer])
            plt.figure(figsize=(3, 3))
            plt.scatter(Y_test_np, y_pred, alpha=0.5)
            plt.plot([Y_test_np.min(), Y_test_np.max()], [Y_test_np.min(), Y_test_np.max()], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Layer {layer}: True vs Predicted Values')
            plt.tight_layout()
            plt.show()
            
    return pipelines, test_scores, test_y, test_pred

def fit_and_evaluate_classification(X_train, Y_train, X_test, Y_test, print_scores=True, make_plot=True):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    X_train_np = [np.array([_x for _x in x]) for x in X_train]
    X_test_np = [np.array([_x for _x in x]) for x in X_test]
    Y_train_np = np.array(Y_train)
    Y_test_np = np.array(Y_test)

    Cs = np.logspace(-3, 3, 10)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluate_fold(X, y, train_idx, val_idx, C):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=C, 
                max_iter=3000,
                class_weight='balanced',  # Add class weighting
                random_state=42
            ))
        ])
        pipeline.fit(X[train_idx], y[train_idx])
        y_val_pred = pipeline.predict(X[val_idx])
        # Use balanced accuracy score instead of regular accuracy
        return balanced_accuracy_score(y[val_idx], y_val_pred)

    pipelines = []
    test_scores = []
    test_y = []
    test_pred = []
    
    for layer in range(len(X_train)):
        # Parallel CV for each C value
        cv_scores = {C: [] for C in Cs}
        for C in Cs:
            scores = Parallel(n_jobs=-1)(
                delayed(evaluate_fold)(
                    X_train_np[layer], Y_train_np, 
                    train_idx, val_idx, C
                )
                for train_idx, val_idx in kf.split(X_train_np[layer])
            )
            cv_scores[C] = np.mean(scores)
        
        # Find best C
        best_C = max(cv_scores.items(), key=lambda x: x[1])[0]
        
        # Train final model with best C
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=best_C, 
                max_iter=3000,
                class_weight='balanced',  # Add class weighting
                random_state=42
            ))
        ])
        pipeline.fit(X_train_np[layer], Y_train_np)
        
        y_train_pred = pipeline.predict(X_train_np[layer])
        y_test_pred = pipeline.predict(X_test_np[layer])
        
        # Use balanced metrics
        train_accuracy = balanced_accuracy_score(Y_train_np, y_train_pred)
        test_accuracy = balanced_accuracy_score(Y_test_np, y_test_pred)
        train_f1 = f1_score(Y_train_np, y_train_pred, average='weighted')
        test_f1 = f1_score(Y_test_np, y_test_pred, average='weighted')
        test_scores.append(Y_test_np==y_test_pred)
        test_y.append(Y_test_np)
        test_pred.append(y_test_pred)

        if print_scores:
            print(f"Layer {layer}:")
            print(f"Best C: {best_C:.3f}")
            print(f"Train Balanced Accuracy: {train_accuracy:.3f}")
            print(f"Test Balanced Accuracy: {test_accuracy:.3f}")
            print(f"Train Weighted F1: {train_f1:.3f}")
            print(f"Test Weighted F1: {test_f1:.3f}")
            # Add class distribution information
            print("Class distribution:")
            for cls in np.unique(Y_train_np):
                print(f"Class {cls}: {np.sum(Y_train_np == cls)} samples")
            print()

        if make_plot:
            # Add confusion matrix visualization
            y_test_pred = pipeline.predict(X_test_np[layer])
            cm = confusion_matrix(Y_test_np, y_test_pred)
            plt.figure(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Layer {layer}: Confusion Matrix')
            plt.tight_layout()
            plt.show()
            
            print()

    return pipelines, test_scores, test_y, test_pred


def fit_and_evaluate_circular_regression(X_train, Y_train, X_test, Y_test, print_scores=True, make_plot=True, figname=None):
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    import numpy as np
    
    # Convert angles to sine and cosine components
    def angle_to_sin_cos(angles):
        return np.column_stack((np.sin(angles), np.cos(angles)))
    
    # Convert sine and cosine predictions back to angles
    def sin_cos_to_angle(sin_vals, cos_vals):
        return np.arctan2(sin_vals, cos_vals)
    
    X_train_np = [np.array([_x for _x in x]) for x in X_train]
    X_test_np = [np.array([_x for _x in x]) for x in X_test]
    Y_train_np = np.array(Y_train)
    Y_test_np = np.array(Y_test)
    
    # Convert target angles to sin/cos components
    Y_train_sin = np.sin(Y_train_np)
    Y_train_cos = np.cos(Y_train_np)
    Y_test_sin = np.sin(Y_test_np)
    Y_test_cos = np.cos(Y_test_np)

    alphas = np.logspace(0, 4, 10)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluate_fold(X, y_sin, y_cos, train_idx, val_idx, alpha):
        # Train sin model
        sin_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        sin_pipeline.fit(X[train_idx], y_sin[train_idx])
        
        # Train cos model
        cos_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        cos_pipeline.fit(X[train_idx], y_cos[train_idx])
        
        # Predict and convert back to angles
        sin_pred = sin_pipeline.predict(X[val_idx])
        cos_pred = cos_pipeline.predict(X[val_idx])
        angle_pred = sin_cos_to_angle(sin_pred, cos_pred)
        
        # Calculate circular error
        true_angles = np.arctan2(y_sin[val_idx], y_cos[val_idx])
        errors = np.abs(np.arctan2(np.sin(angle_pred - true_angles), np.cos(angle_pred - true_angles)))
        mean_circular_error = np.mean(errors)
        
        # Return negative error as score (higher is better)
        return -mean_circular_error

    pipelines = []
    test_scores = []
    test_y = []
    test_pred = []
    
    for layer in range(len(X_train)):
        # Parallel CV for each alpha
        cv_scores = {alpha: [] for alpha in alphas}
        for alpha in alphas:
            scores = Parallel(n_jobs=-1)(
                delayed(evaluate_fold)(
                    X_train_np[layer], Y_train_sin, Y_train_cos, 
                    train_idx, val_idx, alpha
                )
                for train_idx, val_idx in kf.split(X_train_np[layer])
            )
            cv_scores[alpha] = np.mean(scores)
        
        # Find best alpha (lowest error)
        best_alpha = max(cv_scores.items(), key=lambda x: x[1])[0]
        
        # Train final sin model with best alpha
        sin_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=best_alpha))
        ])
        sin_pipeline.fit(X_train_np[layer], Y_train_sin)
        
        # Train final cos model with best alpha
        cos_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=best_alpha))
        ])
        cos_pipeline.fit(X_train_np[layer], Y_train_cos)
        
        # Evaluate on train set
        train_sin_pred = sin_pipeline.predict(X_train_np[layer])
        train_cos_pred = cos_pipeline.predict(X_train_np[layer])
        train_angle_pred = sin_cos_to_angle(train_sin_pred, train_cos_pred)
        train_errors = np.abs(np.arctan2(np.sin(train_angle_pred - Y_train_np), np.cos(train_angle_pred - Y_train_np)))
        train_mean_error = np.mean(train_errors)
        
        # Evaluate on test set
        test_sin_pred = sin_pipeline.predict(X_test_np[layer])
        test_cos_pred = cos_pipeline.predict(X_test_np[layer])
        test_angle_pred = sin_cos_to_angle(test_sin_pred, test_cos_pred)
        test_errors = np.abs(np.arctan2(np.sin(test_angle_pred - Y_test_np), np.cos(test_angle_pred - Y_test_np)))
        test_mean_error = np.mean(test_errors)
        
        pipelines.append((sin_pipeline, cos_pipeline))
        test_scores.append(test_errors)  # Store negative error as score
        test_y.append(Y_test_np)
        test_pred.append(test_angle_pred)
        
        if print_scores:
            print(f"Layer {layer}:")
            print(f"Best alpha: {best_alpha:.3f}")
            print(f"Train mean circular error: {train_mean_error:.3f} radians ({np.degrees(train_mean_error):.1f}°)")
            print(f"Test mean circular error: {test_mean_error:.3f} radians ({np.degrees(test_mean_error):.1f}°)")
            print()

        if make_plot:
            # Also show a scatter plot of true vs predicted
            plt.figure(figsize=(3, 3))
            plt.scatter(Y_test_np, test_angle_pred, alpha=0.5)
            plt.xlabel('True Angle')
            plt.ylabel('Predicted Angle')
            plt.title(f'Layer {layer}: Angle to Goal')
            plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
            plt.yticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
            plt.tight_layout()
            if figname is not None:
                plt.savefig('figs/' + f'buffer_decoding_L{layer}' + figname, dpi=300)
            plt.show()
            
    return pipelines, test_scores, test_y, test_pred