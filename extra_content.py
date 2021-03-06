import copy
from random import shuffle
from file_processing import TestSet, process_string
import perceptron as pt
import decision_tree as dt


def file_main():
    data_set = [['ACD', 0.0231, 1.157, 0.919, 93.061, 0.0917], ['ACD', 0.0296, 1.1183, 0.9356, 80.9492, 0.0681],
                ['ACD', 0.0471, 1.3537, 1.0208, 108.7305, 0.091], ['ACD', 0.0165, 1.2621, 1.1879, 116.3081, 0.1154],
                ['ACD', 0.0236, 1.117, 0.8673, 77.9446, 0.066], ['ACD', 0.008, 1.413, 1.0474, 102.6556, 0.07],
                ['ACD', 0.0267, 1.4068, 1.1244, 107.5716, 0.0734], ['ACD', 0.0838, 1.1258, 1.0406, 100.2574, 0.0474],
                ['ACD', 0.0225, 1.2126, 0.9824, 98.885, 0.0928], ['ACD', 0.0639, 2.1101, 1.2162, 137.5727, 0.159],
                ['ACD', 0.0021, 0.8333, 0.7004, 68.5042, 0.0464], ['ACD', 0.0208, 1.5963, 1.0204, 142.5501, 0.1329],
                ['HM', 0.461, 2.1225, 1.5204, 133.2334, 0.0623], ['HM', 0.2118, 1.5373, 1.2326, 99.011, 0.0808],
                ['HM', 0.2308, 2.3465, 1.3419, 106.459, 0.0548], ['HM', 0.5372, 2.171, 1.8759, 135.6919, 0.0602],
                ['HM', 0.318, 2.1527, 1.1671, 130.0122, 0.0651], ['HM', 0.2434, 2.3092, 1.6817, 179.5259, 0.1192],
                ['HM', 0.4191, 1.5634, 0.8894, 117.2704, 0.0265], ['HM', 0.5952, 2.6538, 1.5957, 152.4041, 0.0752],
                ['HM', 0.3963, 2.0715, 1.2956, 124.8764, 0.094], ['HM', 0.1638, 1.8827, 1.0938, 105.0277, 0.0384],
                ['HM', 0.2752, 3.0803, 1.6789, 146.2936, 0.0803], ['HM', 0.4227, 1.6529, 0.8303, 84.3475, 0.0399]]
    # # decision tree training set
    # data_set = []
    # # Arthur Conan Doyle
    # data_set.append(process("lost_world.txt", "ACD"))
    # data_set.append(process("sherlock.txt", "ACD"))
    # data_set.append(process("study_in_scarlet.txt", "ACD"))
    # data_set.append(process("baskervilles.txt", "ACD"))
    # data_set.append(process("sign_four.txt", "ACD"))
    # data_set.append(process("return.txt", "ACD"))
    # data_set.append(process("memoirs.txt", "ACD"))
    # data_set.append(process("valley.txt", "ACD"))
    # data_set.append(process("tales_terror.txt", "ACD"))
    # data_set.append(process("white_company.txt", "ACD"))
    # data_set.append(process("last_bow.txt", "ACD"))
    # data_set.append(process("boer_war.txt", "ACD"))
    #
    # # Herman Melville
    # data_set.append(process("moby_dick.txt", "HM"))
    # data_set.append(process("bartleby.txt", "HM"))
    # data_set.append(process("confidence_man.txt", "HM"))
    # data_set.append(process("pierre.txt", "HM"))
    # data_set.append(process("white_jacket.txt", "HM"))
    # data_set.append(process("typee.txt", "HM"))
    # data_set.append(process("battle_pieces.txt", "HM"))
    # data_set.append(process("redburn.txt", "HM"))
    # data_set.append(process("omoo.txt", "HM"))
    # data_set.append(process("israel_potter.txt", "HM"))
    # data_set.append(process("my_chimney.txt", "HM"))
    # data_set.append(process("mardi.txt", "HM"))
    # Decision tree test data
    shuffle(data_set)
    # Read test data and process it
    file = open("test_set_file.txt", "r")
    test_data = []
    get_test_data(file, test_data)
    processed_test_data = []
    for data_point in test_data:
        processed_test_data.append(process_string([data_point.author], data_point.data))
    correct_count = 0
    total_count = 0
    # Decision tree classifier usage
    decision_tree = dt.DecisionTree(data_set, ["HM", "ACD"], 4)
    processed_test_data.extend(data_set)
    shuffle(processed_test_data)
    for data_point in processed_test_data:
        total_count += 1
        node = decision_tree.tree
        while node.FINAL_LABEL == "":
            if data_point[node.att_index] <= node.threshold:
                node = node.left
            elif data_point[node.att_index] > node.threshold:
                node = node.right
        if node.FINAL_LABEL == data_point[0]:
            correct_count += 1
    # Perceptron classifier
    processed_test_data.extend(copy.deepcopy(data_set))
    for data_point in data_set:
        if "ACD" in data_point[0]:
            data_point[0] = 1
        elif "HM" in data_point[0]:
            data_point[0] = 0
    for data_point in processed_test_data:
        if "ACD" in data_point[0]:
            data_point[0] = 1
        elif "HM" in data_point[0]:
            data_point[0] = 0
    weights = pt.train_perceptron(processed_test_data, 0.01, 20000)
    for data_point in processed_test_data:
        total_count += 1
        prediction = pt.predict(data_point, weights)
        if data_point[0] == int(prediction):
            correct_count += 1
    print("Perceptron weights: " + str(weights))
    print("Total correct: " + str((correct_count / total_count) * 100))
    print("done")


def get_test_data(file, test_data):
    line = file.readline()
    for i in range(6):
        while "**set_start**" not in line:
            line = file.readline()
        text_data = ""
        data_point = TestSet("", file.readline())
        line = file.readline()
        line = file.readline()

        while "**set_end**" not in line:
            text_data += line
            line = file.readline()
        data_point.data = text_data
        test_data.append(data_point)
