import re

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


class TestSet:
    """
    Object for creating TestSet objects while testing.
    ONLY NEEDED IF TEST SET FILE IS USED
    """
    __slots__ = "data", "author"

    def __init__(self, data, author):
        self.data = data
        self.author = author


def process(file_name, author):
    """
    Process the file and return the dataset row
    :param file_name: The name of the file
    :param author: The name of the author
    :return The dataset row
    """
    dataset_row = [author]
    file = open(file_name, "r")
    result = process_string(dataset_row, file.read())
    return result


def process_string(dataset_row, book_text):
    """
    Process a given string and extract features
    :param dataset_row: The dataset_row in which to add features
    :param book_text: The text to extract from
    :return: The row of the dataset
    """
    # Tokenize sentences using NLTK
    sentences = nltk.sent_tokenize(book_text)

    # Dictionarys for features
    word_dictionary = {}
    operand_dictionary = {}
    hyphenated_words_dictionary = {}
    line_count = 0
    part_speech_dict = {}

    # For each sentence extract the features
    for line in sentences:
        line_count += 1
        words_list = re.findall("[A-Za-z]+-[A-Za-z]+|[A-Za-z]+", line)
        operands_list = re.findall("[^A-Za-z\s]", line)
        hyphenated_words_list = re.findall("[A-Za-z]+-[A-Za-z]+", line)
        part_speech = nltk.pos_tag(nltk.word_tokenize(line))

        for part in part_speech:
            key = part[1]
            if key in part_speech_dict.keys():
                part_speech_dict[key] = part_speech_dict[key] + 1
            else:
                part_speech_dict[key] = 1

        for word in words_list:
            if word.lower() in word_dictionary.keys():
                word_dictionary[word.lower()] = word_dictionary[word.lower()] + 1
            else:
                word_dictionary[word.lower()] = 1

        for hyp in hyphenated_words_list:
            if hyp.lower() in hyphenated_words_dictionary.keys():
                hyphenated_words_dictionary[hyp.lower()] = hyphenated_words_dictionary[hyp.lower()] + 1
            else:
                hyphenated_words_dictionary[hyp.lower()] = 1

        for operand in operands_list:
            if operand in operand_dictionary.keys():
                operand_dictionary[operand] = operand_dictionary[operand] + 1
            else:
                operand_dictionary[operand] = 1
    # Sort the dictionaries and do the calculations
    sorted_word_dictionary = sorted(word_dictionary.items(), key=lambda kv: kv[1], reverse=True)
    sorted_operand_dictionary = sorted(operand_dictionary.items(), key=lambda kv: kv[1], reverse=True)
    banned_list = set(stopwords.words('english'))
    sorted_word_dictionary_copy = sorted_word_dictionary.copy()
    for i in range(0, len(sorted_word_dictionary_copy)):
        if sorted_word_dictionary_copy[i][0] in banned_list:
            sorted_word_dictionary.remove(sorted_word_dictionary_copy[i])
    banned_operands = []
    sorted_operand_dictionary_copy = sorted_operand_dictionary.copy()
    for i in range(0, len(sorted_operand_dictionary_copy)):
        if sorted_operand_dictionary_copy[i][0] in banned_operands:
            sorted_operand_dictionary.remove(sorted_operand_dictionary_copy[i])

    sorted_sentences = sorted(sentences, key=len)
    mean_length_sentence = 0
    for sentence in sorted_sentences:
        mean_length_sentence += len(sentence)

    mean_word_dictionary = sorted(word_dictionary.items(), key=lambda kv: len(kv[0]))
    mean = 0
    for word in mean_word_dictionary:
        mean += len(word[0])

    # Get the value of each feature and add it to the row
    if ";" in operand_dictionary.keys():
        dataset_row.append(round(operand_dictionary[";"] / line_count, 4))
    else:
        dataset_row.append(0)

    if "," in operand_dictionary.keys():
        dataset_row.append(round(operand_dictionary[","] / line_count, 4))
    else:
        dataset_row.append(0)

    if "RB" in part_speech_dict.keys():
        dataset_row.append(round(part_speech_dict["RB"] / line_count, 4))
    else:
        dataset_row.append(0)

    dataset_row.append(round(mean_length_sentence / len(sorted_sentences), 4))

    if "-" in operand_dictionary.keys():
        dataset_row.append(round(operand_dictionary["-"] / line_count, 4))
    else:
        dataset_row.append(0)

    if "captain" in word_dictionary.keys():
        dataset_row.append(round(word_dictionary["captain"] / line_count, 4))
    else:
        dataset_row.append(0)

    if "holmes" in word_dictionary.keys():
        dataset_row.append(round(word_dictionary["holmes"] / line_count, 4))
    else:
        dataset_row.append(0)

    if "ship" in word_dictionary.keys():
        dataset_row.append(round(word_dictionary["ship"] / line_count, 4))
    else:
        dataset_row.append(0)

    return dataset_row
