import copy
import pickle
import sys
from random import shuffle

import perceptron as pt
import decision_tree as dt
import file_processing as fp


def main():
    """
    The main function which does the prediction and training process
    :return: None
    """
    # The data set that was created from the books of the authors "ACD" stands for Arthur Conan Doyle and "HM" stands
    # for Herman Melville
    data_set = [['ACD', 0.0231, 1.157, 0.919, 93.061, 0.3164, 0.0002, 0, 0.0002],
                ['ACD', 0.0296, 1.1183, 0.9356, 80.9492, 0.1673, 0.0003, 0.0676, 0.0012],
                ['ACD', 0.0471, 1.3537, 1.0208, 108.7305, 0.2409, 0, 0.0439, 0.0005],
                ['ACD', 0.0165, 1.2621, 1.1879, 116.3081, 0.2096, 0, 0.071, 0.0007],
                ['ACD', 0.0236, 1.117, 0.8673, 77.9446, 0.2274, 0.0055, 0.0462, 0.0007],
                ['ACD', 0.008, 1.413, 1.0474, 102.6556, 0.2672, 0.0031, 0.1281, 0.0015],
                ['ACD', 0.0267, 1.4068, 1.1244, 107.5716, 0.2203, 0.0014, 0.0782, 0.0016],
                ['ACD', 0.0838, 1.1258, 1.0406, 100.2574, 0.2593, 0.0052, 0.0491, 0.0003],
                ['ACD', 0.0225, 1.2126, 0.9824, 98.885, 0.2537, 0.0015, 0, 0.0007],
                ['ACD', 0.0639, 2.1101, 1.2162, 137.5727, 0.3374, 0.002, 0, 0.0043],
                ['ACD', 0.0021, 0.8333, 0.7004, 68.5042, 0.2152, 0, 0.0464, 0],
                ['ACD', 0.0208, 1.5963, 1.0204, 142.5501, 0.2363, 0.0088, 0, 0.0002],
                ['HM', 0.461, 2.1225, 1.5204, 133.2334, 0.2841, 0.0359, 0, 0.052],
                ['HM', 0.2118, 1.5373, 1.2326, 99.011, 0.4051, 0, 0, 0],
                ['HM', 0.2308, 2.3465, 1.3419, 106.459, 0.5562, 0.0022, 0, 0.0014],
                ['HM', 0.5372, 2.171, 1.8759, 135.6919, 0.8562, 0.0012, 0, 0.0008],
                ['HM', 0.318, 2.1527, 1.1671, 130.0122, 0.9746, 0.0651, 0, 0.0513],
                ['HM', 0.2434, 2.3092, 1.6817, 179.5259, 0.5633, 0.0102, 0.0006, 0.0179],
                ['HM', 0.4191, 1.5634, 0.8894, 117.2704, 1.2626, 0.0084, 0, 0.0039],
                ['HM', 0.5952, 2.6538, 1.5957, 152.4041, 0.5649, 0.0387, 0, 0.0493],
                ['HM', 0.3963, 2.0715, 1.2956, 124.8764, 0.4177, 0.032, 0, 0.0344],
                ['HM', 0.1638, 1.8827, 1.0938, 105.0277, 0.4157, 0.0241, 0, 0.023],
                ['HM', 0.2752, 3.0803, 1.6789, 146.2936, 0.6697, 0.0069, 0, 0],
                ['HM', 0.4227, 1.6529, 0.8303, 84.3475, 0.4739, 0.0001, 0, 0.0006]]

    if len(sys.argv) > 1:
        # if neither arguments are given then exit with warning
        if sys.argv[1] != "train" and sys.argv[1] != "predict":
            print("Unknown argument, please enter 'predict' or 'train'")
            sys.exit(1)

        # Train
        elif sys.argv[1] == "train":
            # Train your model
            model = input("Which model would you like to train ? Perceptron(p) or Decision Tree(d): ")
            if model != "p" and model != "d":
                print("Sorry! Wrong argument")

            # Ask user if they want to use perceptron
            elif model == "p":
                perceptron_data = copy.deepcopy(data_set)
                for data_point in perceptron_data:
                    if "ACD" in data_point[0]:
                        data_point[0] = 1
                    elif "HM" in data_point[0]:
                        data_point[0] = 0
                shuffle(perceptron_data)
                weights = pt.train_perceptron(perceptron_data, 0.01, 20000)
                predict = input("A perceptron has been trained. Would you like to make a prediction?(y/n) ")
                if predict == "y":
                    filename = input(
                        "Please enter the name of the file containing text for author identification: ")
                    data_value = fp.process(filename, "NA")
                    prediction = pt.predict(data_value, weights)
                    if int(prediction) == 1:
                        print("Author is Arthur Conan Doyle.")
                    elif int(prediction) == 0:
                        print("Author is Herman Melville.")

            # Or a decision tree
            elif model == "d":
                max_depth = int(input("Please enter the maximum depth of the decision tree: "))
                entropy_cutoff = float(
                    input("Please enter the entropy cutoff of the decision tree(ideal is 0.0): "))
                print("Training a decision tree on training data...")
                shuffle(data_set)
                tree = dt.DecisionTree(data_set, ["ACD", "HM"], max_depth, entropy_cutoff)

                predict = input("The decision tree has been trained. Would you like to make a prediction?(y/n) ")
                if predict == "y":
                    filename = input(
                        "Please enter the name of the file containing text for author identification: ")
                    data_value = fp.process(filename, "NA")

                    node = tree.tree
                    while node.FINAL_LABEL == "":
                        if data_value[node.att_index] <= node.threshold:
                            node = node.left
                        elif data_value[node.att_index] > node.threshold:
                            node = node.right
                    if node.FINAL_LABEL == "ACD":
                        print("The author is Arthur Conan Doyle")
                    else:
                        print("The author is Herman Melville")

        # Use hard coded models for prediction
        elif sys.argv[1] == "predict":
            # if user explicitly mentions the need of a decision tree then use a decision tree
            if sys.argv[2] == '-d':
                with open('model_decision_tree.pkl', 'rb') as model_input:
                    tree = pickle.load(model_input)
                filename = sys.argv[3]
                print("Predicting using the hard-coded decision tree")
                file_data = fp.process(filename, "NA")
                node = tree.tree
                while node.FINAL_LABEL == "":
                    if file_data[node.att_index] <= node.threshold:
                        node = node.left
                    elif file_data[node.att_index] > node.threshold:
                        node = node.right
                if node.FINAL_LABEL == "ACD":
                    print("The author is Arthur Conan Doyle")
                else:
                    print("The author is Herman Melville")
            # Otherwise use a perceptron
            else:
                filename = sys.argv[2]
                print("Predicting using the hard-coded perceptron, to predict using the hard-coded decision tree use "
                      "'predict -d filename' ")
                model_file = open("model_perceptron.txt", "r")
                line = model_file.readline().split(",")
                weights = []
                for weight in line:
                    weights.append(float(weight))
                data_value = fp.process(filename, "NA")
                prediction = pt.predict(data_value, weights)
                if int(prediction) == 1:
                    print("Author is Arthur Conan Doyle.")
                elif int(prediction) == 0:
                    print("Author is Herman Melville")

    else:
        # Exit if no argument given
        print("Please enter argument 'train' or 'predict'. ")
        sys.exit(1)


if __name__ == '__main__':
    main()
