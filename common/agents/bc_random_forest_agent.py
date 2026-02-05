"""
Behavioral Cloning Agent using Random Forest.

A robust ensemble method that uses sklearn's RandomForestClassifier
for policy learning via behavioral cloning.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from common.agents.agent import Agent


class BCRandomForestAgent(Agent):
    """
    Behavioral Cloning agent using a Random Forest classifier.

    Random Forests are ensemble methods that combine multiple decision trees
    to improve generalization and reduce overfitting.
    """

    def __init__(self, state_dimension: int, number_of_actions: int,
                 n_estimators: int = 100, max_depth: int = None,
                 min_samples_leaf: int = 1):
        """Initialize BC Random Forest Agent.

        Args:
            state_dimension (int): State dimension
            number_of_actions (int): Number of actions
            n_estimators (int, optional): Number of trees. Defaults to 100.
            max_depth (int, optional): Maximum depth of trees. None for unlimited.
            min_samples_leaf (int, optional): Minimum samples per leaf. Defaults to 1.
        """
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 1, accuracy_threshold: float = 100.0):
        """
        Train the random forest on behavioral cloning data.

        Note: epochs parameter is ignored for random forests (single fit).

        Args:
            env: Environment (used for feature names)
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Ignored for random forests

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

        # Fit the random forest
        print(f"\nTraining Random Forest (n_estimators={self.n_estimators}, "
              f"max_depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf})...")
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
        print(f"Random Forest Training Complete!")
        print(f"{'='*70}")
        print(f"Forest Statistics:")
        print(f"  Number of trees:    {self.n_estimators}")
        print(f"  Max depth:          {self.max_depth}")
        print(f"  Train Accuracy:     {train_accuracy:.4f}% ({n_correct}/{n_total} correct, {n_wrong} wrong)")
        if test_accuracy is not None:
            print(f"  Test Accuracy:      {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return 0, train_accuracy, test_accuracy, 0.0, 0.0

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using the trained random forest.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): Ignored for random forests. Defaults to False.

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
        """Save the random forest to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        joblib.dump(self.classifier, 'tmp_model/bc_random_forest.joblib')

        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, 'tmp_model/bc_random_forest_meta.joblib')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the random forest from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            classifier_path = os.path.join(model_root_folder_path, 'bc_random_forest.joblib')
            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
                self.is_trained = True
                print(f"Successfully loaded Random Forest agent from {classifier_path}")

                meta_path = os.path.join(model_root_folder_path, 'bc_random_forest_meta.joblib')
                if os.path.exists(meta_path):
                    metadata = joblib.load(meta_path)
                    self.n_estimators = metadata.get('n_estimators', self.n_estimators)
                    self.max_depth = metadata.get('max_depth', self.max_depth)
                    self.min_samples_leaf = metadata.get('min_samples_leaf', self.min_samples_leaf)
            else:
                print(f"Warning: Random forest file not found at {classifier_path}")
        except Exception as msg:
            print(f"Error loading Random Forest agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }
