import spacy

nlp = spacy.load("en_core_web_sm")


def extract_noun_phrases(caption):
    doc = nlp(caption.lower())
    phrases = []

    for chunk in doc.noun_chunks:
        if any(token.pos_ == "NOUN" for token in chunk):
            phrases.append(chunk.text)

    return phrases