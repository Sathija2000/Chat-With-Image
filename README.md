# Chat-With-Image

An AI-powered Visual Question Answering tool that allows users to ask natural language questions about uploaded images. Built with **LangChain**, **OpenAI GPT**, **HuggingFace vision models (BLIP & DETR)**, and **Streamlit**.

---

##  Features

-  Upload any `.jpg`, `.jpeg`, or `.png` image
-  Automatically generates a caption using **BLIP**
-  Detects objects in the image using **DETR**
-  Ask natural language questions, and get answers powered by **GPT-3.5**
-  LangChain tools and memory for contextual conversation

---

##  Tech Stack

- Python
- [Streamlit](https://streamlit.io/) – Web UI
- [LangChain](https://www.langchain.com/) – Tool orchestration
- [HuggingFace Transformers](https://huggingface.co/) – BLIP & DETR
- [OpenAI GPT-3.5](https://platform.openai.com/docs) – Language model
- [PIL](https://pillow.readthedocs.io/) – Image processing
- Torch – Inference engine

---

##  Installation

1. **Clone the repo**:

```bash
git clone https://github.com/your-username/chat-with-image.git
cd chat-with-image
