import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from common.interpreter.interpreter import *

class DecisionTreeInterpreter(Interpreter):

    def __init__(self, config):
        super().__init__(config)

    def interpret(self, env, rl_agent, model_checking_info):
        X = np.array(model_checking_info['collected_states'])
        y = np.array(model_checking_info['collected_action_idizes'])

        action_labels = env.action_mapper.actions

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # Stack train and test data together
        X = np.vstack((X_train, X_test))
        y = np.hstack((y_train, y_test))
        print(X.shape, y.shape)

        # Train a decision tree classifier
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)

        # Make predictions using the trained classifier
        y_pred = clf.predict(X)

        # Print the accuracy of the classifier
        acc = accuracy_score(y_pred, y)
        print("Interpreter Accuracy:", acc)

        # Plot the decision tree using Matplotlib
        fig, ax = plt.subplots(figsize=(30,15), dpi=300)

        # Use compressed feature names if available, otherwise use regular feature names
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            feature_names = state_mapper.get_compressed_feature_names()
            if feature_names is None:
                feature_names = state_mapper.get_feature_names()
        else:
            feature_names = state_mapper.get_feature_names()

        tree.plot_tree(clf, filled=True, class_names=action_labels, feature_names=feature_names)
        plt.title("Decision Tree")
        plt.savefig('interpretion_plot.png')
