import operator
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def process(fileName):
    print("**************** Processing file: " + fileName + "****************")
    file = open(fileName, "r")
    wordDictionary = {}
    operandDictionary = {}
    hyphenatedWordsDictionary = {}
    lineCount = 0
    partSpeechDict = {}
    for line in file:
        lineCount += 1
        wordsList = re.findall("[A-Za-z]+-[A-Za-z]+|[A-Za-z]+", line)
        operandsList = re.findall("[^A-Za-z\s]", line)
        hyphenatedWordsList = re.findall("[A-Za-z]+-[A-Za-z]+", line)
        partSpeech = nltk.pos_tag(nltk.word_tokenize(line))

        for part in partSpeech:
            key = part[1]
            if key in partSpeechDict.keys():
                partSpeechDict[key] = partSpeechDict[key] + 1
            else:
                partSpeechDict[key] = 1

        for word in wordsList:
            if word in wordDictionary.keys():
                wordDictionary[word] = wordDictionary[word] + 1
            else:
                wordDictionary[word] = 1

        for hyp in hyphenatedWordsList:
            if hyp in hyphenatedWordsDictionary.keys():
                hyphenatedWordsDictionary[hyp] = hyphenatedWordsDictionary[hyp] + 1
            else:
                hyphenatedWordsDictionary[hyp] = 1

        for operand in operandsList:
            if operand in operandDictionary.keys():
                operandDictionary[operand] = operandDictionary[operand] + 1
            else:
                operandDictionary[operand] = 1

    sorted_word_dictionary = sorted(wordDictionary.items(), key=lambda kv: kv[1], reverse=True)
    sorted_operand_dictionary = sorted(operandDictionary.items(), key=lambda kv: kv[1], reverse=True)
    sorted_part_dictionary = sorted(partSpeechDict.items(), key=lambda kv: kv[1], reverse=True)

    bannedList = []

    sorted_word_dictionary_copy = sorted_word_dictionary.copy()
    for i in range(0, len(sorted_word_dictionary_copy)):
        if sorted_word_dictionary_copy[i][0] in bannedList:
            sorted_word_dictionary.remove(sorted_word_dictionary_copy[i])

    bannedOperands = []

    sorted_operand_dictionary_copy = sorted_operand_dictionary.copy()
    for i in range(0, len(sorted_operand_dictionary_copy)):
        if sorted_operand_dictionary_copy[i][0] in bannedOperands:
            sorted_operand_dictionary.remove(sorted_operand_dictionary_copy[i])

    mean_word_dictionary = sorted(wordDictionary.items(), key=lambda kv: len(kv[0]))
    median_tuple = mean_word_dictionary[len(mean_word_dictionary) // 2]
    print("Median word length: " + str(len(median_tuple[0])))
    print("Median word : " + median_tuple[0])

    mean = 0
    for word in mean_word_dictionary:
        mean += len(word[0])
    print("Mean word length: " + str(mean // len(mean_word_dictionary)))

    words_file = open(fileName + "word_results.txt", 'w')
    for tuple in sorted_word_dictionary:
        words_file.write(tuple[0] + " : " + str(tuple[1]) + " : " + str(round(tuple[1] / lineCount, 4)) + "\n\n")

    words_file = open(fileName + "operand_results.txt", 'w')
    for tuple in sorted_operand_dictionary:
        words_file.write(tuple[0] + " : " + str(tuple[1]) + " : " + str(round(tuple[1] / lineCount, 4)) + "\n\n")

    words_file = open(fileName + "parts_of_speech_result.txt", 'w')
    for tuple in sorted_part_dictionary:
        words_file.write(tuple[0] + " : " + str(tuple[1]) + " : " + str(round(tuple[1] / lineCount, 4)) + "\n\n")

    hypWordCount = len(hyphenatedWordsDictionary.keys())
    print("hyphenated words : " + str(hypWordCount) + " : " + str(round(hypWordCount / lineCount, 4)))

    print("**************** Ending file process: " + fileName + "****************")


process("lost_world.txt")
process("sherlock.txt")
process("study_in_scarlet.txt")
process("moby_dick.txt")
process("bartleby.txt")
process("confidence_man.txt")