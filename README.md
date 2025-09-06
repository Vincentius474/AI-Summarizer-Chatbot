# AI Summarizer Chatbot

An interactive **Streamlit** app that lets you upload lecture slide PDFs, automatically extract text and images, summarize slide contents with **transformer-based models**, and perform **question answering** using FAISS-powered semantic search.

---

## Features

- **PDF Processing**  
  - Extracts text and images from each slide/page.  
  - Displays content in an easy-to-navigate UI.  

- **AI-Powered Summarization**  
  - Generates concise summaries of each slide using `facebook/bart-large-cnn`.  
  - Option to summarize the entire deck at once.  

- **Question Answering**  
  - Builds a semantic FAISS index of slide content with `SentenceTransformer`.  
  - Allows you to ask questions and get context-aware answers.  

- **Interactive User Interface**  
  - Sidebar navigation for selecting slides.  
  - Inline display of extracted images.  
  - Instant answers to queries about the uploaded lecture material.  

---

## Tech Stack

- [Streamlit](https://streamlit.io/) – Web app framework  
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) – PDF text and image extraction  
- [Pillow (PIL)](https://pillow.readthedocs.io/) – Image processing  
- [Transformers](https://huggingface.co/transformers/) – Summarization & Q&A models  
- [SentenceTransformers](https://www.sbert.net/) – Text embeddings  
- [FAISS](https://faiss.ai/) – Semantic search and similarity search  
- [NumPy](https://numpy.org/) – Numerical operations  

---

## Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/ai-summarizer-chatbot.git
   cd ai-summarizer-chatbot
