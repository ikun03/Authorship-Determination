import decision_tree_helper


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
            entropy_2 = decision_tree_helper.entropy(labels, list2)
            entropy_1 = decision_tree_helper.entropy(labels, list1)
            step_entropy.append(
                max(entropy_1, entropy_2))
    step_entropy.sort()
    return step_entropy[0]


class DecisionTree:
    class Node:
        __slots__ = "values", "left", "right", "stumps", "attribute_index", "threshold", "data_label"

        def __init__(self, values, stumps, attribute_index, threshold):
            self.values = values
            self.stumps = stumps
            self.left = []
            self.right = []
            self.attribute_index = attribute_index
            self.threshold = threshold
            self.data_label = ""

        def append_to_left(self, element):
            self.left.append(element)

        def append_to_right(self, element):
            self.right.append(element)

    __slots__ = "data", "root"

    def __init__(self, data, labels):
        self.data = data
        attribute_threshold = {}
        for i in range(1, len(data[0])):
            attribute_threshold[i] = calculate_threshold(data, i, labels)
