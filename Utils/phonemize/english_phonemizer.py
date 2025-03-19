"""
English Phonemizer
-----------------
This module provides functions for converting English text to phonemes using a more comprehensive rule-based approach.
"""

import re

# More comprehensive English phoneme mappings
VOWEL_PHONEME_MAP = {
    'a': 'æ',  # as in "cat"
    'ai': 'eɪ',  # as in "day"
    'ay': 'eɪ',  # as in "day"
    'ar': 'ɑɹ',  # as in "car"
    'au': 'ɔ',  # as in "caught"
    'aw': 'ɔ',  # as in "law"
    'e': 'ɛ',  # as in "bed"
    'ee': 'i',  # as in "see"
    'ea': 'i',  # as in "sea"
    'er': 'ɝ',  # as in "her"
    'i': 'ɪ',  # as in "sit"
    'ie': 'aɪ',  # as in "tie"
    'ir': 'ɝ',  # as in "bird"
    'o': 'ɑ',  # as in "hot"
    'oo': 'u',  # as in "food"
    'ou': 'aʊ',  # as in "out"
    'ow': 'aʊ',  # as in "how"
    'oi': 'ɔɪ',  # as in "boy"
    'oy': 'ɔɪ',  # as in "toy"
    'or': 'ɔɹ',  # as in "for"
    'u': 'ʌ',  # as in "but"
    'ur': 'ɝ',  # as in "fur"
}

CONSONANT_PHONEME_MAP = {
    'b': 'b',
    'c': 'k',  # Default to 'k' sound
    'ch': 'tʃ',  # as in "chair"
    'd': 'd',
    'dg': 'dʒ',  # as in "edge"
    'f': 'f',
    'g': 'g',  # Default to hard 'g'
    'h': 'h',
    'j': 'dʒ',  # as in "jump"
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',  # as in "sing"
    'p': 'p',
    'ph': 'f',  # as in "phone"
    'q': 'kw',  # as in "queen"
    'r': 'ɹ',
    's': 's',
    'sh': 'ʃ',  # as in "ship"
    't': 't',
    'th': 'θ',  # as in "thin" (default to unvoiced)
    'v': 'v',
    'w': 'w',
    'wh': 'w',  # as in "what"
    'x': 'ks',  # as in "box"
    'y': 'j',  # as in "yes"
    'z': 'z',
    'zh': 'ʒ',  # as in "vision"
}

# Special cases for common words
SPECIAL_CASES = {
    'the': 'ðə',
    'a': 'ə',
    'an': 'ən',
    'and': 'ænd',
    'is': 'ɪz',
    'are': 'ɑɹ',
    'was': 'wʌz',
    'were': 'wɝ',
    'of': 'ʌv',
    'to': 'tu',
    'in': 'ɪn',
    'for': 'fɔɹ',
    'on': 'ɑn',
    'at': 'æt',
    'by': 'baɪ',
    'with': 'wɪθ',
    'from': 'fɹʌm',
    'my': 'maɪ',
    'your': 'jɔɹ',
    'his': 'hɪz',
    'her': 'hɝ',
    'their': 'ðɛɹ',
    'our': 'aʊɹ',
    'hello': 'hɛloʊ',
    'hi': 'haɪ',
    'name': 'neɪm',
    'kadir': 'kədiɹ',
}

def clean_text(text):
    """Clean up the text by removing unnecessary characters and normalizing spaces."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text

def english_phonemize(text):
    """
    Convert English text to a more comprehensive phonetic representation.
    
    Args:
        text (str): English text to phonemize
        
    Returns:
        str: Phonemized text
    """
    # Clean the text
    text = clean_text(text)
    
    # Add spaces around punctuation to preserve them
    text = re.sub(r'([,.!?;:])', r' \1 ', text)
    
    # Split into words
    words = text.split()
    
    phonemized_words = []
    for word in words:
        # Skip punctuation
        if word in ',.!?;:':
            phonemized_words.append(word)
            continue
            
        # Check if the word is a special case
        if word in SPECIAL_CASES:
            phonemized_words.append(SPECIAL_CASES[word])
            continue
            
        # Apply phoneme mappings
        phonemized_word = word
        
        # First apply vowel mappings (longer patterns first)
        vowel_patterns = sorted(VOWEL_PHONEME_MAP.keys(), key=len, reverse=True)
        for pattern in vowel_patterns:
            phonemized_word = phonemized_word.replace(pattern, VOWEL_PHONEME_MAP[pattern])
            
        # Then apply consonant mappings (longer patterns first)
        consonant_patterns = sorted(CONSONANT_PHONEME_MAP.keys(), key=len, reverse=True)
        for pattern in consonant_patterns:
            phonemized_word = phonemized_word.replace(pattern, CONSONANT_PHONEME_MAP[pattern])
            
        phonemized_words.append(phonemized_word)
    
    # Join the phonemized words
    phonemized_text = ' '.join(phonemized_words)
    
    # Clean up spaces around punctuation
    phonemized_text = re.sub(r'\s+([,.!?;:])', r'\1', phonemized_text)
    phonemized_text = re.sub(r'([,.!?;:])\s+', r'\1 ', phonemized_text)
    
    return phonemized_text
