import streamlit as st
from pipeline import run_pipeline
from PIL import Image
import tempfile

st.title("AI Image Labeling System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        image.save(tmp.name)

        with st.spinner("Processing..."):
            results = run_pipeline(tmp.name)

    st.subheader("Generated Caption")
    st.write(results["caption"])

    st.subheader("Extracted Phrases")
    st.write(results["phrases"])

    st.subheader("Best Label")
    st.write(results["best_label"])

    st.subheader("Phrase Rankings")
    st.write(results["ranking"])