"""
Behavioral Cloning Agent using K-Nearest Neighbors.

A simple, non-parametric agent that uses sklearn's KNeighborsClassifier
for policy learning via behavioral cloning.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from common.agents.agent import Agent


class BCKNNAgent(Agent):
    """
    Behavioral Cloning agent using K-Nearest Neighbors classifier.

    KNN is a non-parametric method that classifies based on the majority
    vote of the k nearest neighbors in the training data.
    """

    def __init__(self, state_dimension: int, number_of_actions: int,
                 n_neighbors: int = 5, weights: str = 'uniform',
                 metric: str = 'euclidean'):
        """Initialize BC KNN Agent.

        Args:
            state_dimension (int): State dimension
            number_of_actions (int): Number of actions
            n_neighbors (int, optional): Number of neighbors. Defaults to 5.
            weights (str, optional): Weight function ('uniform' or 'distance'). Defaults to 'uniform'.
            metric (str, optional): Distance metric. Defaults to 'euclidean'.
        """
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 1, accuracy_threshold: float = 100.0):
        """
        Train the KNN on behavioral cloning data.

        Note: epochs parameter is ignored for KNN (single fit).

        Args:
            env: Environment (used for feature names)
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Ignored for KNN

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

        # Scale features for better KNN performance
        print(f"\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit the KNN
        print(f"\nTraining KNN (n_neighbors={self.n_neighbors}, "
              f"weights={self.weights}, metric={self.metric})...")
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
        print(f"KNN Training Complete!")
        print(f"{'='*70}")
        print(f"Model Statistics:")
        print(f"  K (neighbors):      {self.n_neighbors}")
        print(f"  Weights:            {self.weights}")
        print(f"  Metric:             {self.metric}")
        print(f"  Training samples:   {len(X_train)}")
        print(f"  Train Accuracy:     {train_accuracy:.4f}% ({n_correct}/{n_total} correct, {n_wrong} wrong)")
        if test_accuracy is not None:
            print(f"  Test Accuracy:      {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return 0, train_accuracy, test_accuracy, 0.0, 0.0

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using the trained KNN.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): Ignored for KNN. Defaults to False.

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
        """Save the KNN to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        joblib.dump(self.classifier, 'tmp_model/bc_knn.joblib')
        joblib.dump(self.scaler, 'tmp_model/bc_knn_scaler.joblib')

        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, 'tmp_model/bc_knn_meta.joblib')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the KNN from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            classifier_path = os.path.join(model_root_folder_path, 'bc_knn.joblib')
            scaler_path = os.path.join(model_root_folder_path, 'bc_knn_scaler.joblib')

            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                print(f"Successfully loaded KNN agent from {classifier_path}")

                meta_path = os.path.join(model_root_folder_path, 'bc_knn_meta.joblib')
                if os.path.exists(meta_path):
                    metadata = joblib.load(meta_path)
                    self.n_neighbors = metadata.get('n_neighbors', self.n_neighbors)
                    self.weights = metadata.get('weights', self.weights)
                    self.metric = metadata.get('metric', self.metric)
            else:
                print(f"Warning: KNN file not found at {classifier_path}")
        except Exception as msg:
            print(f"Error loading KNN agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric
        }
