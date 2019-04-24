from math import log2


def entropy_prod(total_length, values_length):
    ratio = total_length / values_length
    return ratio * (log2(ratio))


def information_gain(main_list, labels, attribute_index, threshold_value):
    list1 = []
    list2 = []
    for value in range(0, len(main_list)):
        if main_list[value][attribute_index] <= threshold_value:
            list1.append(value)
        else:
            list2.append(value)

    parentEntropy = entropy(labels, main_list)
    list1Entropy = entropy(labels, list1)
    list2Entropy = entropy(labels, list2)


def entropy(labels, main_list):
    label_list = []
    for i in range(len(labels)):
        label_list.append([])
    for index in range(len(main_list)):
        value_ind = labels.index(main_list[0])
        label_list[value_ind].append(index)
    entropy_parent = 0
    for list in label_list:
        entropy_parent += entropy_prod(len(main_list), len(list))
    return -1 * entropy_parent
