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


def process_tree_node(decision_tree):
    pass


class DecisionTree:
    class Node:
        __slots__ = "values", "left", "right", "stumps", "attribute_index", "threshold", "data_labels", "attrib_thresholds"

        def __init__(self, values, stumps, attribute_index, threshold, attribute_thresholds):
            self.values = values
            self.stumps = stumps
            self.left = []
            self.right = []
            self.attribute_index = attribute_index
            self.threshold = threshold
            self.data_labels = ""
            self.attrib_thresholds = attribute_thresholds

        def append_to_left(self, element):
            self.left.append(element)

        def append_to_right(self, element):
            self.right.append(element)

    __slots__ = "data", "root"

    def __init__(self, data, labels):
        self.data = data
        # Calculate thresholds
        attribute_threshold = {}
        for i in range(1, len(data[0])):
            attribute_threshold[i] = calculate_threshold(data, i, labels)
        max_info_gain = [0, 0]

        # Get the attribute with max info gain that shall be our root
        for att_ind in range(1, len(data[0])):
            info_gain = dth.information_gain(data, labels, att_ind, attribute_threshold[att_ind])
            if info_gain > max_info_gain[1]:
                max_info_gain[0] = att_ind
                max_info_gain[1] = info_gain

        # Get the stumps that the child will get
        stumps = []
        for att_ind in range(1, len(data[0])):
            if max_info_gain[0] != att_ind:
                stumps.append(att_ind)

        # Creating the root node
        decision_tree = DecisionTree.Node(data, stumps, max_info_gain[0], attribute_threshold[max_info_gain[0]],
                                          attribute_threshold)
        process_tree_node(decision_tree)
