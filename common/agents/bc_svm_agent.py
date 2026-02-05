"""
Behavioral Cloning Agent using Support Vector Machine.

A powerful classifier that uses sklearn's SVC for policy learning
via behavioral cloning.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from common.agents.agent import Agent


class BCSVMAgent(Agent):
    """
    Behavioral Cloning agent using Support Vector Machine classifier.

    SVMs are effective for high-dimensional spaces and work well when
    the number of dimensions exceeds the number of samples.
    """

    def __init__(self, state_dimension: int, number_of_actions: int,
                 C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale'):
        """Initialize BC SVM Agent.

        Args:
            state_dimension (int): State dimension
            number_of_actions (int): Number of actions
            C (float, optional): Regularization parameter. Defaults to 1.0.
            kernel (str, optional): Kernel type ('linear', 'rbf', 'poly'). Defaults to 'rbf'.
            gamma (str, optional): Kernel coefficient. Defaults to 'scale'.
        """
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        self.classifier = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 1, accuracy_threshold: float = 100.0):
        """
        Train the SVM on behavioral cloning data.

        Note: epochs parameter is ignored for SVMs (single fit).

        Args:
            env: Environment (used for feature names)
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Ignored for SVMs

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

        # Scale features for better SVM performance
        print(f"\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit the SVM
        print(f"\nTraining SVM (C={self.C}, kernel={self.kernel}, gamma={self.gamma})...")
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Compute training accuracy
        y_train_pred = self.classifier.predict(X_train_scaled)
        train_accuracy = 100 * accuracy_score(y_train, y_train_pred)
        n_correct = (y_train_pred == y_train).sum()
        n_total = len(y_train)
        n_wrong = n_total - n_correct

        # Compute test accuracy if available
        test_accuracy = None
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            y_test_pred = self.classifier.predict(X_test_scaled)
            test_accuracy = 100 * accuracy_score(y_test, y_test_pred)

        # Print statistics
        print(f"\n{'='*70}")
        print(f"SVM Training Complete!")
        print(f"{'='*70}")
        print(f"Model Statistics:")
        print(f"  Kernel:             {self.kernel}")
        print(f"  C:                  {self.C}")
        print(f"  Gamma:              {self.gamma}")
        print(f"  Support vectors:    {len(self.classifier.support_vectors_)}")
        print(f"  Train Accuracy:     {train_accuracy:.4f}% ({n_correct}/{n_total} correct, {n_wrong} wrong)")
        if test_accuracy is not None:
            print(f"  Test Accuracy:      {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return 0, train_accuracy, test_accuracy, 0.0, 0.0

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using the trained SVM.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): Ignored for SVMs. Defaults to False.

        Returns:
            int: Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.number_of_actions)

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        state_scaled = self.scaler.transform(state_2d)
        action = self.classifier.predict(state_scaled)[0]
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
        state_scaled = self.scaler.transform(state_2d)
        return self.classifier.predict_proba(state_scaled)[0]

    def save(self, artifact_path='model'):
        """Save the SVM to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        joblib.dump(self.classifier, 'tmp_model/bc_svm.joblib')
        joblib.dump(self.scaler, 'tmp_model/bc_svm_scaler.joblib')

        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, 'tmp_model/bc_svm_meta.joblib')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the SVM from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            classifier_path = os.path.join(model_root_folder_path, 'bc_svm.joblib')
            scaler_path = os.path.join(model_root_folder_path, 'bc_svm_scaler.joblib')

            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                print(f"Successfully loaded SVM agent from {classifier_path}")

                meta_path = os.path.join(model_root_folder_path, 'bc_svm_meta.joblib')
                if os.path.exists(meta_path):
                    metadata = joblib.load(meta_path)
                    self.C = metadata.get('C', self.C)
                    self.kernel = metadata.get('kernel', self.kernel)
                    self.gamma = metadata.get('gamma', self.gamma)
            else:
                print(f"Warning: SVM file not found at {classifier_path}")
        except Exception as msg:
            print(f"Error loading SVM agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'C': self.C,
            'kernel': self.kernel,
            'gamma': self.gamma
        }
