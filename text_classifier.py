import copy
import pickle
import sys
from random import shuffle

import perceptron as pt
import decision_tree as dt
import file_processing as fp


def main():
    data_set = [['ACD', 0.0231, 1.157, 0.919, 93.061, 0.3164], ['ACD', 0.0296, 1.1183, 0.9356, 80.9492, 0.1673],
                ['ACD', 0.0471, 1.3537, 1.0208, 108.7305, 0.2409], ['ACD', 0.0165, 1.2621, 1.1879, 116.3081, 0.2096],
                ['ACD', 0.0236, 1.117, 0.8673, 77.9446, 0.2274], ['ACD', 0.008, 1.413, 1.0474, 102.6556, 0.2672],
                ['ACD', 0.0267, 1.4068, 1.1244, 107.5716, 0.2203], ['ACD', 0.0838, 1.1258, 1.0406, 100.2574, 0.2593],
                ['ACD', 0.0225, 1.2126, 0.9824, 98.885, 0.2537], ['ACD', 0.0639, 2.1101, 1.2162, 137.5727, 0.3374],
                ['ACD', 0.0021, 0.8333, 0.7004, 68.5042, 0.2152], ['ACD', 0.0208, 1.5963, 1.0204, 142.5501, 0.2363],
                ['HM', 0.461, 2.1225, 1.5204, 133.2334, 0.2841], ['HM', 0.2118, 1.5373, 1.2326, 99.011, 0.4051],
                ['HM', 0.2308, 2.3465, 1.3419, 106.459, 0.5562], ['HM', 0.5372, 2.171, 1.8759, 135.6919, 0.8562],
                ['HM', 0.318, 2.1527, 1.1671, 130.0122, 0.9746], ['HM', 0.2434, 2.3092, 1.6817, 179.5259, 0.5633],
                ['HM', 0.4191, 1.5634, 0.8894, 117.2704, 1.2626], ['HM', 0.5952, 2.6538, 1.5957, 152.4041, 0.5649],
                ['HM', 0.3963, 2.0715, 1.2956, 124.8764, 0.4177], ['HM', 0.1638, 1.8827, 1.0938, 105.0277, 0.4157],
                ['HM', 0.2752, 3.0803, 1.6789, 146.2936, 0.6697], ['HM', 0.4227, 1.6529, 0.8303, 84.3475, 0.4739]]

    if len(sys.argv) > 1:
        if sys.argv[1] != "train" and sys.argv[1] != "predict":
            print("Unknown argument, please enter 'predict' or 'train'")
            sys.exit(1)

        elif sys.argv[1] == "train":
            # Train your model
            model = input("Which model would you like to train ? Perceptron(p) or Decision Tree(d): ")
            if model != "p" and model != "d":
                print("Sorry! Wrong argument")

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
                    filename = input("Please enter the name of the file containing text for author identification: ")
                    data_value = fp.process(filename, "NA")
                    prediction = pt.predict(data_value, weights)
                    if int(prediction) == 1:
                        print("Author is Arthur Conan Doyle.")
                    elif int(prediction) == 0:
                        print("Author is Herman Melville.")
            elif model == "d":
                max_depth = int(input("Please enter the maximum depth of the decision tree: "))
                entropy_cutoff = float(input("Please enter the entropy cutoff of the decision tree(ideal is 0.0): "))
                print("Training a decision tree on training data...")
                tree = dt.DecisionTree(shuffle(data_set), ["ACD", "HM"], max_depth, entropy_cutoff)

                predict = input("The decision tree has been trained. Would you like to make a prediction?(y/n) ")
                if predict == "y":
                    filename = input("Please enter the name of the file containing text for author identification: ")
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

        elif sys.argv[1] == "predict":
            if sys.argv[2] == '-d':
                tree = pickle.load("model_decision_tree.pkl")
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
        print("Please enter argument 'train' or 'predict'. ")
        sys.exit(1)


if __name__ == '__main__':
    main()
