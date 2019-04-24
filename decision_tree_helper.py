from math import log2


def entropy_prod(total_length, values_length):
    ratio = values_length / total_length
    if ratio == 0:
        return 0
    return ratio * (log2(ratio))


def information_gain(main_list, labels, attribute_index, threshold_value):
    list1 = []
    list2 = []
    for value in range(0, len(main_list)):
        if main_list[value][attribute_index] <= threshold_value:
            list1.append(main_list[value])
        else:
            list2.append(main_list[value])

    parent_entropy = entropy(labels, main_list)
    list1_entropy = entropy(labels, list1)
    list2_entropy = entropy(labels, list2)
    weighted_avg_1 = len(list1) / len(main_list)
    weighted_avg_2 = len(list2) / len(main_list)
    sum_weighted_entropy = (list1_entropy * weighted_avg_1) + (list2_entropy * weighted_avg_2)
    info_gain = parent_entropy - sum_weighted_entropy
    return info_gain


def entropy(labels, main_list):
    label_list = []
    for i in range(len(labels)):
        label_list.append([])
    for ind in range(len(main_list)):
        value_ind = labels.index(main_list[ind][0])
        label_list[value_ind].append(ind)
    entropy_parent = 0
    for list in label_list:
        entropy_parent += entropy_prod(len(main_list), len(list))
    return -1 * entropy_parent
