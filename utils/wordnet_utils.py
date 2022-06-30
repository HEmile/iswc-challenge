from nltk.corpus import wordnet as wn
import nltk
import random

nltk.download('wordnet')
nltk.download('omw-1.4')


def distort_sentence(sentence: str) -> str:
    words = sentence.split(" ")
    for i in range(5):
      replace_index = random.randint(0, len(words) -2)
      word = words[replace_index]
      synonyms = wn.synsets(word)
      if len(synonyms) > 0:
        synonym = synonyms[0].lemmas()[0].name()
        words[replace_index] = synonym
    return " ".join(words)


test_sentence = """
Which countries share a border with Morocco?

['sahara', 'western sahara', 'mauritania', 'algeria', 'spain']"""

print(distort_sentence(test_sentence))
