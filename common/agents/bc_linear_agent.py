"""
Behavioral Cloning Agent using Linear Functions (Logistic Regression).

A simple, interpretable agent that uses sklearn's LogisticRegression
for policy learning via behavioral cloning.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from common.agents.agent import Agent


class BCLinearAgent(Agent):
    """
    Behavioral Cloning agent using Logistic Regression (linear classifier).

    This agent learns a linear decision boundary in the feature space,
    making it highly interpretable and efficient.
    """

    def __init__(self, state_dimension: int, number_of_actions: int,
                 C: float = 1.0, max_iter: int = 1000):
        """Initialize BC Linear Agent.

        Args:
            state_dimension (int): State dimension
            number_of_actions (int): Number of actions
            C (float, optional): Inverse regularization strength. Defaults to 1.0.
            max_iter (int, optional): Maximum iterations for solver. Defaults to 1000.
        """
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.C = C
        self.max_iter = max_iter

        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1
        )
        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 1, accuracy_threshold: float = 100.0):
        """
        Train the logistic regression on behavioral cloning data.

        Note: epochs parameter is ignored (single fit).

        Args:
            env: Environment (used for feature names)
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Ignored for logistic regression

        Returns:
            Tuple of (epoch, train_acc, test_acc, train_loss, test_loss)
        """
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_test = data.get('X_test')
        y_test = data.get('y_test')

        if X_train is None or y_train is None:
            return None, None, None, None, None

        # Print class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass distribution in training data:")
        for cls, count in zip(unique, counts):
            print(f"  Action {cls}: {count} samples")

        # Fit the logistic regression
        print(f"\nTraining Logistic Regression (C={self.C}, max_iter={self.max_iter})...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Compute training accuracy
        y_train_pred = self.classifier.predict(X_train)
        train_accuracy = 100 * accuracy_score(y_train, y_train_pred)
        n_correct = (y_train_pred == y_train).sum()
        n_total = len(y_train)
        n_wrong = n_total - n_correct

        # Compute test accuracy if available
        test_accuracy = None
        if X_test is not None and y_test is not None:
            y_test_pred = self.classifier.predict(X_test)
            test_accuracy = 100 * accuracy_score(y_test, y_test_pred)

        # Print statistics
        print(f"\n{'='*70}")
        print(f"Logistic Regression Training Complete!")
        print(f"{'='*70}")
        print(f"Model Statistics:")
        print(f"  Regularization (C): {self.C}")
        print(f"  Number of classes:  {len(self.classifier.classes_)}")
        print(f"  Train Accuracy:     {train_accuracy:.4f}% ({n_correct}/{n_total} correct, {n_wrong} wrong)")
        if test_accuracy is not None:
            print(f"  Test Accuracy:      {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return 0, train_accuracy, test_accuracy, 0.0, 0.0

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using the trained logistic regression.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): Ignored. Defaults to False.

        Returns:
            int: Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.number_of_actions)

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        action = self.classifier.predict(state_2d)[0]
        return int(action)

    def get_raw_outputs(self, state: np.ndarray) -> np.ndarray:
        """Get class probabilities for the given state.

        Args:
            state (np.ndarray): Current state

        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_trained:
            return np.ones(self.number_of_actions) / self.number_of_actions

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        return self.classifier.predict_proba(state_2d)[0]

    def save(self, artifact_path='model'):
        """Save the logistic regression to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        joblib.dump(self.classifier, 'tmp_model/bc_linear.joblib')

        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'C': self.C,
            'max_iter': self.max_iter,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, 'tmp_model/bc_linear_meta.joblib')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the logistic regression from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            classifier_path = os.path.join(model_root_folder_path, 'bc_linear.joblib')
            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
                self.is_trained = True
                print(f"Successfully loaded Linear agent from {classifier_path}")

                meta_path = os.path.join(model_root_folder_path, 'bc_linear_meta.joblib')
                if os.path.exists(meta_path):
                    metadata = joblib.load(meta_path)
                    self.C = metadata.get('C', self.C)
                    self.max_iter = metadata.get('max_iter', self.max_iter)
            else:
                print(f"Warning: Linear model file not found at {classifier_path}")
        except Exception as msg:
            print(f"Error loading Linear agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'C': self.C,
            'max_iter': self.max_iter
        }

    def get_coefficients(self):
        """Get the learned coefficients (weights) for interpretability.

        Returns:
            dict: Coefficients and intercepts for each class
        """
        if not self.is_trained:
            return None

        return {
            'coefficients': self.classifier.coef_,
            'intercept': self.classifier.intercept_,
            'classes': self.classifier.classes_
        }
