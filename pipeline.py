from models.segmentation import segment_and_crop
from models.captioning import generate_caption
from models.nlp import extract_noun_phrases
from models.clip_ranker import rank_phrases_with_clip


def run_pipeline(image_path):

    cropped = segment_and_crop(image_path)
    caption = generate_caption(cropped)
    phrases = extract_noun_phrases(caption)
    best_phrase, ranking = rank_phrases_with_clip(cropped, phrases)

    return {
        "caption": caption,
        "phrases": phrases,
        "best_label": best_phrase,
        "ranking": ranking
    }