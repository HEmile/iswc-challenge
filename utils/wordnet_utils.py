from nltk.corpus import wordnet as wn
import nltk
import random


# nltk.download('wordnet')
# nltk.download('omw-1.4')


def distort_sentence(sentence: str) -> str:
    """
        Distorts the input sentence with by replacing random tokens with synonyms.
    :param sentence:
    :return:
    """
    words = sentence.split(" ")
    for i in range(5):
        replace_index = random.randint(0, len(words) - 2)
        word = words[replace_index]
        synonyms = wn.synsets(word)
        if len(synonyms) > 0:
            synonym = synonyms[0].lemmas()[0].name()
            words[replace_index] = synonym
    return " ".join(words)


def augment_sentence(sentence: str) -> str:
    """
        WIP: Maybe we will switch this component out for HuggingFace paraphrasing.

        Augments the input sentence with Hypernymy and Hyponymy information.
        Might need some kind of NER to find nouns...
    :param sentence:
    :return:
    """
    words = sentence.split(" ")
    for i in range(5):
        replace_index = random.randint(0, len(words) - 2)
        word = words[replace_index]

        for synset in wn.synsets(word):
            for hyper in synset.hypernyms():
                print(synset, hyper)
                words[replace_index] = hyper
    return " ".join(words)


test_sentence = """Which countries share a border with Morocco?"""

# Uncomment to test
    # print(distort_sentence(test_sentence))
# print(augment_sentence(test_sentence))