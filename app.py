import io
import fitz
import faiss
import numpy as np
from PIL import Image
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer

"""
Load models
"""
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
embedder = load_embedder()

def extract_text_and_images(pdf_file):
    """
    PDF Extractor
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    slides = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        image_list = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(io.BytesIO(img_bytes))
            image_list.append(image)

        slides.append({
            "page": page_num + 1,
            "text": text.strip(),
            "images": image_list
        })
    return slides

def summarize_text(text):
    """
    Summarization
    """
    if not text.strip():
        return "No text found on this slide."
    try:
        input_length = len(text.split())
        max_len = min(120, input_length + 20)
        summary = summarizer(text, max_length=max_len, min_length=10, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error summarizing: {e}"

def build_faiss_index(slides):
    """
    FAISS Index Builder
    """
    texts = [s["text"] for s in slides if s["text"].strip()]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, texts

def answer_question(query, index, texts, k=3):
    """
    Q&A Function
    """
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    context = "\n\n".join([texts[i] for i in I[0]])
    try:
        response = summarizer(
            f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            max_length=150,
            min_length=30,
            do_sample=False
        )
        return response[0]["summary_text"]
    except Exception as e:
        return f"Error generating answer: {e}"

"""
Streamlit User Interface
"""
st.set_page_config(page_title="AI Lecture Assistant", layout="wide")

st.title("AI Lecture Slide Assistant")
st.write("Upload a PDF of lecture slides to get summaries, view images, and ask questions.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting content..."):
        slides = extract_text_and_images(uploaded_file)

    st.success(f"Extracted {len(slides)} slides.")

    """ Sidebar navigation """
    slide_nums = [s["page"] for s in slides]
    selected_slide = st.sidebar.selectbox("Select Slide", slide_nums)

    """ Display selected slide """
    slide = slides[selected_slide - 1]
    st.subheader(f"Slide {slide['page']}")

    """ Extracted text """
    st.write("### Extracted Text")
    st.text(slide["text"] if slide["text"] else "No text detected.")

    """ AI summary """
    st.write("### AI Summary")
    summary = summarize_text(slide["text"])
    st.info(summary)

    """ Extracted images """
    if slide["images"]:
        st.write("### Extracted Images")
        for idx, img in enumerate(slide["images"]):
            resized_img = img.resize((400, 300))
            st.image(resized_img, caption=f"Slide {slide['page']} - Image {idx+1}")
    else:
        st.write("No images detected on this slide.")

    """ For summarizing the entire deck """
    if st.button("Summarize Entire Deck"):
        with st.spinner("Summarizing all slides..."):
            all_summaries = [
                {"page": s["page"], "summary": summarize_text(s["text"])}
                for s in slides
            ]
        st.success("Summarization complete!")
        for s in all_summaries:
            st.markdown(f"**Slide {s['page']}**: {s['summary']}")

    """ Q & A Section """
    st.write("### Ask Questions About the Slides")
    if "faiss_index" not in st.session_state:
        with st.spinner("Building knowledge base..."):
            index, texts = build_faiss_index(slides)
            st.session_state.faiss_index = index
            st.session_state.texts = texts
        st.success("Knowledge base ready!")

    query = st.text_input("Ask a question about the lecture:")
    if query:
        with st.spinner("Thinking..."):
            answer = answer_question(query, st.session_state.faiss_index, st.session_state.texts)
        st.write("**Answer:**")
        st.success(answer)
