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
    min_ind = 0
    for i in range(len(step_entropy)):
        if step_entropy[i] < step_entropy[min_ind]:
            min_ind = i

    return label_means[0] + min_ind * step


def process_tree_node(data, decision_tree_node):
    # Check if entropy is 0 or there are no stumps to pass
    node_entropy = dth.entropy(decision_tree_node.labels, data)
    if node_entropy == 0:
        decision_tree_node.FINAL_LABEL = decision_tree_node.labels[0]
        return decision_tree_node
    if len(decision_tree_node.stumps) == 0:
        label_dictionary = {}
        for label in decision_tree_node.labels:
            label_dictionary[label] = 0
        for data_point in data:
            label_dictionary[data_point[0]] += 1
        max_count = 0
        for label in label_dictionary.keys():
            val = label_dictionary[label]
            if val > max_count:
                max_count = val
                decision_tree_node.FINAL_LABEL = label
        return decision_tree_node
    # Split data to node
    left_node_data = []
    right_node_data = []
    for data_point in data:
        if data_point[decision_tree_node.att_index] <= decision_tree_node.threshold:
            left_node_data.append(data_point)
        else:
            right_node_data.append(data_point)

    # For left node find the stump with max info gain

    left_node_labels = []
    for data_point in left_node_data:
        if data_point[0] not in left_node_labels:
            left_node_labels.append(data_point[0])

    left_max_info_gain = [0, -1]
    for att_ind in range(1, len(left_node_data[0])):
        info_gain = dth.information_gain(left_node_data, left_node_labels, att_ind,
                                         decision_tree_node.threshold_list[att_ind])
        if info_gain > left_max_info_gain[1]:
            left_max_info_gain[0] = att_ind
            left_max_info_gain[1] = info_gain
    left_node_stumps = decision_tree_node.stumps
    if left_max_info_gain[0] in left_node_stumps:
        left_node_stumps.remove(left_max_info_gain[0])

    left_node = DecisionTree.Node(left_node_stumps, left_max_info_gain[0],
                                  decision_tree_node.threshold_list[left_max_info_gain[0]],
                                  decision_tree_node.threshold_list)
    left_node.labels = left_node_labels
    left_node = process_tree_node(left_node_data, left_node)
    decision_tree_node.left = left_node

    # For right node find the stump with max info gain

    right_node_labels = []
    for data_point in right_node_data:
        if data_point[0] not in right_node_labels:
            right_node_labels.append(data_point[0])

    right_max_info_gain = [0, -1]
    for att_ind in range(1, len(right_node_data[0])):
        info_gain = dth.information_gain(right_node_data, right_node_labels, att_ind,
                                         decision_tree_node.threshold_list[att_ind])
        if info_gain > right_max_info_gain[1]:
            right_max_info_gain[0] = att_ind
            right_max_info_gain[1] = info_gain
    right_node_stumps = decision_tree_node.stumps
    if right_max_info_gain[0] in right_node_stumps:
        right_node_stumps.remove(right_max_info_gain[0])

    right_node = DecisionTree.Node(right_node_stumps, right_max_info_gain[0],
                                   decision_tree_node.threshold_list[right_max_info_gain[0]],
                                   decision_tree_node.threshold_list)
    right_node.labels = right_node_labels
    right_node = process_tree_node(right_node_data, right_node)
    decision_tree_node.right = right_node

    return decision_tree_node


class DecisionTree:
    __slots__ = "tree"

    def __init__(self, data, labels):
        # Calculate thresholds
        threshold_list = {}
        for i in range(1, len(data[0])):
            threshold_list[i] = calculate_threshold(data, i, labels)

        # Get the attribute with max info gain that shall be our root
        max_info_gain = [0, -1]
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
        decision_tree = DecisionTree.Node(stumps, max_info_gain[0], threshold_list[max_info_gain[0]],
                                          threshold_list)
        decision_tree.labels = labels
        self.tree = process_tree_node(data, decision_tree)

    class Node:
        __slots__ = "left", "right", "stumps", "att_index", "threshold", "labels", "threshold_list", "FINAL_LABEL"

        def __init__(self, stumps, attribute_index, threshold, attribute_thresholds):
            self.stumps = stumps
            self.left = None
            self.right = None
            self.att_index = attribute_index
            self.threshold = threshold
            self.labels = ""
            self.threshold_list = attribute_thresholds
            self.FINAL_LABEL = ""

        def append_to_left(self, element):
            self.left.append(element)

        def append_to_right(self, element):
            self.right.append(element)


data = [["slow", 35, 0.4, 40],
        ["slow", 35, 0.1, 40],
        ["fast", 5, 0.4, 150],
        ["fast", 35, 0.1, 150]]
labels = ["slow", "fast"]
tree = DecisionTree(data, labels)
print("Should be done")
