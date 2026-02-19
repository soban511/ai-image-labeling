# AI Image Labeling System

This project provides an AI-powered image labeling and segmentation system using deep learning and NLP. It combines U²-Net for segmentation, BLIP for image captioning, and CLIP for phrase ranking, with a Streamlit web interface.

## Features

- **Automatic Image Segmentation**: Uses U²-Net for salient object detection and cropping.
- **Image Captioning**: Generates descriptive captions with BLIP.
- **Noun Phrase Extraction**: Extracts key phrases from captions using spaCy.
- **Phrase Ranking**: Ranks phrases by relevance to the image using CLIP.
- **Web Interface**: User-friendly Streamlit app for uploading and labeling images.

## Folder Structure

```
segmenter.py
pipeline.py
app.py
requirements.txt
output/
models/
  segmentation.py
  captioning.py
  nlp.py
  clip_ranker.py
U-2-Net/
  data_loader.py
  model/
    u2net.py
    u2net_refactor.py
  u2net.pth
  requirements.txt
  ... (other scripts, test data, figures, gradio demo, etc.)
```

## Main Components

- **segmenter.py**: Loads U²-Net and BLIP models, provides segmentation and captioning utilities.
- **pipeline.py**: Orchestrates the full pipeline: segmentation → captioning → phrase extraction → ranking.
- **models/**: Modular code for segmentation, captioning, NLP, and CLIP ranking.
- **app.py**: Streamlit web app for interactive use.
- **U-2-Net/**: Contains the U²-Net model, weights, and related scripts.

## Installation

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
   For U²-Net-specific dependencies:
   ```
   pip install -r U-2-Net/requirements.txt
   ```

2. Download the U²-Net model weights (`u2net.pth`) and place them in `U-2-Net/`.

## Usage

- **Web App**:
  ```
  streamlit run app.py
  ```
  Upload an image to get a caption, best label, and segmentation.

- **Pipeline**:
  Use `run_pipeline(image_path)` from `pipeline.py` for programmatic access.

## Credits

- U²-Net: [Official Repo](https://github.com/xuebinqin/U-2-Net)
- BLIP: Salesforce
- CLIP: OpenAI

---
