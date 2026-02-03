"""
Behavioral Cloning Agent using Decision Tree.

A simple, interpretable agent that uses sklearn's DecisionTreeClassifier
for policy learning via behavioral cloning.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from common.agents.agent import Agent


class BCDecisionTreeAgent(Agent):
    """
    Behavioral Cloning agent using a Decision Tree classifier.

    This agent is fully interpretable and can be visualized as a decision tree.
    """

    def __init__(self, state_dimension: int, number_of_actions: int,
                 max_depth: int = None, min_samples_leaf: int = 1):
        """Initialize BC Decision Tree Agent.

        Args:
            state_dimension (int): State dimension
            number_of_actions (int): Number of actions
            max_depth (int, optional): Maximum depth of tree. None for unlimited.
            min_samples_leaf (int, optional): Minimum samples per leaf. Defaults to 1.
        """
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        self.classifier = tree.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 1):
        """
        Train the decision tree on behavioral cloning data.

        Note: epochs parameter is ignored for decision trees (single fit).

        Args:
            env: Environment (used for feature names in visualization)
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Ignored for decision trees

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

        # Fit the decision tree
        print(f"\nTraining Decision Tree (max_depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf})...")
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

        # Print tree statistics
        print(f"\n{'='*70}")
        print(f"Decision Tree Training Complete!")
        print(f"{'='*70}")
        print(f"Tree Statistics:")
        print(f"  Depth:          {self.classifier.get_depth()}")
        print(f"  Number of leaves: {self.classifier.get_n_leaves()}")
        print(f"  Train Accuracy: {train_accuracy:.4f}% ({n_correct}/{n_total} correct, {n_wrong} wrong)")
        if test_accuracy is not None:
            print(f"  Test Accuracy:  {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        # Decision trees don't have a loss function in the traditional sense
        # Return 0.0 for loss to maintain interface compatibility
        return 0, train_accuracy, test_accuracy, 0.0, 0.0

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using the trained decision tree.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): Ignored for decision trees. Defaults to False.

        Returns:
            int: Selected action
        """
        if not self.is_trained:
            # Return random action if not trained
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
        """Save the decision tree to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        # Save the classifier
        joblib.dump(self.classifier, 'tmp_model/bc_decision_tree.joblib')

        # Save metadata
        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, 'tmp_model/bc_decision_tree_meta.joblib')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the decision tree from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            classifier_path = os.path.join(model_root_folder_path, 'bc_decision_tree.joblib')
            if os.path.exists(classifier_path):
                self.classifier = joblib.load(classifier_path)
                self.is_trained = True
                print(f"Successfully loaded Decision Tree agent from {classifier_path}")

                # Load metadata if available
                meta_path = os.path.join(model_root_folder_path, 'bc_decision_tree_meta.joblib')
                if os.path.exists(meta_path):
                    metadata = joblib.load(meta_path)
                    self.max_depth = metadata.get('max_depth', self.max_depth)
                    self.min_samples_leaf = metadata.get('min_samples_leaf', self.min_samples_leaf)
            else:
                print(f"Warning: Decision tree file not found at {classifier_path}")
        except Exception as msg:
            print(f"Error loading Decision Tree agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }

    def export_tree_visualization(self, env, filename='decision_tree.png'):
        """Export a visualization of the decision tree.

        Args:
            env: Environment (for feature and action names)
            filename: Output filename
        """
        if not self.is_trained:
            print("Cannot visualize: tree not trained yet")
            return

        import matplotlib.pyplot as plt

        # Get feature names from environment
        try:
            feature_names = env.storm_bridge.state_mapper.get_feature_names()
        except:
            feature_names = [f"feature_{i}" for i in range(self.state_dimension)]

        # Get action names
        try:
            action_names = env.action_mapper.actions
        except:
            action_names = [f"action_{i}" for i in range(self.number_of_actions)]

        plt.figure(figsize=(20, 10))
        tree.plot_tree(
            self.classifier,
            filled=True,
            feature_names=feature_names,
            class_names=action_names,
            rounded=True
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Decision tree visualization saved to {filename}")
