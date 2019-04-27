Files included:
text_classifier.py - Main file to run for the classifier
decision_tree.py - Implementation of the decision tree
decision_tree_helper.py - Helper functions for the decision tree
model_decision_tree.pkl - The file for the decision object to be used by pickle in python
model-perceptron.txt - The weights for the hardcoded trained model of the perceptron

Running:
In order to run the program simply run text_classifier.py with either 'train' or 'predict'.
train will ask you for arguments and configurations as you go along.
predict filename - will by default perform a perceptron classifier for determining accuracy
predict -d filename - will use the hardcoded decision tree instead of perceptron