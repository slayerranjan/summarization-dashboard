import re
import math
from collections import Counter
import nltk

# Ensure punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def compression_ratio(original_text: str, summary_text: str) -> float:
    """Calculate compression ratio: how much shorter the summary is compared to the original."""
    orig_len = len(original_text.split())
    sum_len = len(summary_text.split())
    if orig_len == 0:
        return 0.0
    return round(((orig_len - sum_len) / orig_len) * 100, 2)

def readability_score(text: str) -> float:
    """Calculate Flesch-Kincaid grade level for readability."""
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    syllables = sum(count_syllables(word) for word in words)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)

    # Flesch-Kincaid Grade Level formula
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
    return round(grade, 2)

def count_syllables(word: str) -> int:
    """Rough syllable counter using vowels."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)

def entity_retention(original_text: str, summary_text: str) -> float:
    """Check how many named entities from original are retained in summary."""
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree

    def get_entities(text):
        chunks = ne_chunk(pos_tag(word_tokenize(text)))
        entities = []
        for chunk in chunks:
            if isinstance(chunk, Tree):
                entity = " ".join(c[0] for c in chunk.leaves())
                entities.append(entity)
        return set(entities)

    orig_entities = get_entities(original_text)
    sum_entities = get_entities(summary_text)

    if not orig_entities:
        return 0.0

    retained = len(orig_entities.intersection(sum_entities))
    return round((retained / len(orig_entities)) * 100, 2)
