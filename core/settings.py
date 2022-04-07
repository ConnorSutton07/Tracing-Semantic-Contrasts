import nltk

stopwords = set(nltk.corpus.stopwords.words('english') +
    ["ye", "thy", "thee", "hast", "chorus", "strophe", "antistrophe", "thou", "pg", "o'er", "chor", "hath", "0", "thine", "chapter", "twas", "said", "would", "could", "upon", "shall", "like"])