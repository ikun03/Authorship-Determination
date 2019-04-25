import decision_tree_helper as dth


def calculate_threshold(data, i, labels):
    labels_dict = {}
    for label in labels:
        labels_dict[label] = 0
        labels_dict[str(label) + "_count"] = 0
    for value in data:
        labels_dict[value[0]] += value[i]
        labels_dict[str(value[0]) + "_count"] += 1
    label_means = []
    for label in labels:
        if label in labels_dict.keys():
            mean = labels_dict[label] / labels_dict[str(label) + "_count"]
            label_means.append(mean)
    label_means.sort()
    step_range = label_means[-1] - label_means[0]
    step = step_range / 5
    step_entropy = []
    for index in range(5):
        list1 = []
        list2 = []
        threshold = label_means[0] + (index * step)
        for value in data:
            if value[i] <= threshold:
                list1.append(value)
            else:
                list2.append(value)
            entropy_2 = dth.entropy(labels, list2)
            entropy_1 = dth.entropy(labels, list1)
            step_entropy.append(
                max(entropy_1, entropy_2))
    step_entropy.sort()
    return step_entropy[0]


def process_tree_node(decision_tree_node):
    # Check if entropy is 0 or there are no stumps to pass
    node_entropy = dth.entropy(decision_tree_node.labels, decision_tree_node.data)
    if node_entropy == 0:
        decision_tree_node.FINAL_LABEL = decision_tree_node.labels[0]
        return
    if len(decision_tree_node.stumps) == 0:
        label_dictionary = {}
        for label in decision_tree_node.labels:
            label_dictionary[label] = 0
        for data_point in decision_tree_node.data:
            label_dictionary[data_point[0]] += 1
        max_count = 0
        for label in label_dictionary.keys():
            val = label_dictionary[label]
            if val > max_count:
                max_count = val
                decision_tree_node.FINAL_LABEL = label
        return
    # Split data to node
    left_node_data = []
    right_node_data = []
    for data_point in decision_tree_node.data:
        if data_point[decision_tree_node.att_index] <= decision_tree_node.threshold:
            left_node_data.append(data_point)
        else:
            right_node_data.append(data_point)


class DecisionTree:
    __slots__ = "data", "root"

    def __init__(self, data, labels):
        self.data = data
        # Calculate thresholds
        threshold_list = {}
        for i in range(1, len(data[0])):
            threshold_list[i] = calculate_threshold(data, i, labels)
        max_info_gain = [0, 0]

        # Get the attribute with max info gain that shall be our root
        for att_ind in range(1, len(data[0])):
            info_gain = dth.information_gain(data, labels, att_ind, threshold_list[att_ind])
            if info_gain > max_info_gain[1]:
                max_info_gain[0] = att_ind
                max_info_gain[1] = info_gain

        # Get the stumps that the child will get
        stumps = []
        for att_ind in range(1, len(data[0])):
            if max_info_gain[0] != att_ind:
                stumps.append(att_ind)

        # Creating the root node
        decision_tree = DecisionTree.Node(data, stumps, max_info_gain[0], threshold_list[max_info_gain[0]],
                                          threshold_list)
        process_tree_node(decision_tree)

    class Node:
        __slots__ = "data", "left", "right", "stumps", "att_index", "threshold", "labels", "threshold_list", "FINAL_LABEL"

        def __init__(self, values, stumps, attribute_index, threshold, attribute_thresholds):
            self.data = values
            self.stumps = stumps
            self.left = []
            self.right = []
            self.att_index = attribute_index
            self.threshold = threshold
            self.labels = ""
            self.threshold_list = attribute_thresholds
            self.FINAL_LABEL = ""

        def append_to_left(self, element):
            self.left.append(element)

        def append_to_right(self, element):
            self.right.append(element)
