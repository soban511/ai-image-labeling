import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="openai"
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")


def rank_phrases_with_clip(image_np, phrases):

    if len(phrases) == 0:
        return None, []

    image = Image.fromarray(image_np)
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_emb = clip_model.encode_image(image_tensor)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    text_tokens = clip_tokenizer(phrases).to(device)

    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    similarity = (image_emb @ text_emb.T).squeeze(0)
    scores = similarity.cpu().numpy()

    phrase_scores = list(zip(phrases, scores))
    phrase_scores_sorted = sorted(
        phrase_scores,
        key=lambda x: x[1],
        reverse=True
    )

    best_phrase = phrase_scores_sorted[0][0]

    return best_phrase, phrase_scores_sorted