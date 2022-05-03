import nltk

stopwords = set(nltk.corpus.stopwords.words('english') +
    ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0", "thine", "chapter", "twas", "said", "would", "could", "upon", "shall", "like"])

'''
key_words = {
    'x': [
        '1',
        '2', 
        '3', 
    ],
    'y': [
        '1',
        '2', 
        '3', 
    ],
    'z': [
        '1',
        '2', 
        '3', 
    ]
}
'''

key_words = [
        'battle',
        'warrior',
        'king',
        'beowulf',
        'war',
        'man',
        'woman',
        'sword',
        'grendel',
        'hero',
        'good',
        'evil'
    ]
