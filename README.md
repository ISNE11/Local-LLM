# Local-LLM
A simple way to run local Large Language Models (LLMs) offline using [Ollama](https://ollama.com/).

---
## 🛠️ Install Ollama

Ollama makes it super easy to run models locally.

1. Go to 👉 [https://ollama.com/download](https://ollama.com/download)
2. Download and install for your OS (Windows, macOS, or Linux).


---

## 📁 Project Structure & File Contents


```
README.md                # This file. Project overview and instructions.
main.py                  # Main GUI app: chat, textbook toggle, question type, stop button.
run_llm.py               # (Legacy) Script to interact with the local LLM via CLI.
textbook_llm.py          # (Legacy) Script for querying the textbook using the LLM via CLI.
textbook.txt             # Text version of the textbook used for context.
db/                      # Database folder for storing embeddings and data.
   chroma.sqlite3         # SQLite database for vector storage (Chroma).
   a77b493b-.../          # Chroma DB internal files.
textbook_create/         # Utilities for creating the textbook text file.
   pdf_to_text.py         # Script to convert PDF textbook to text.
   textbook-pdf/          # Folder to store original textbook PDFs.
```

- **main.py**: Main entry point. Launches a GUI chat app with:
  - Toggle for using textbook context (on by default)
  - Toggle for open-ended or multiple choice questions
  - Button to stop the AI response at any time
  - Real-time streaming, markdown rendering, and scrollable chat
- **run_llm.py**: (Legacy) CLI for running LLM queries.
- **textbook_llm.py**: (Legacy) CLI for textbook-based Q&A.
- **textbook.txt**: The context source for textbook-based Q&A.
- **db/**: Stores vector database files for fast retrieval.
- **textbook_create/**: Tools for converting and managing textbook files.


## 🧠 Run a Model

After installing Ollama, open your terminal and run:

```sh
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
## or which ever model you want to use
```

You’ll now have a local chatbot that works entirely offline.

---

## 🚀 Setup & Usage Guide

1. **Clone this repository**
   ```sh
   git clone https://github.com/ISNE11/Local-LLM.git
   cd Local-LLM
   ```

3. **Install Python dependencies**

   ```sh
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
   ```sh
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install any needed packages manually as you run the scripts.)*

4. **Start Ollama**
   - Make sure Ollama is running and the desired model is available.
   - Make sure the python files are set to use the correct model name.


5. **Run the main GUI app**
   ```sh
   python main.py
   ```

   - Use the toggles at the top to enable/disable textbook context and switch between open-ended or multiple choice questions.
   - Click the **Stop AI** button at any time to abort a long response.
   - All chat is streamed in real time with markdown rendering and scrollable history.

6. **(Optional) Run legacy CLI scripts**
   ```sh
   python run_llm.py
   # or
   python textbook_llm.py
   ```

7. **Interact with your local LLM!**

---

## 📄 Notes
- This repo assumes you have Python 3.8+ installed.
- Ollama must be running in the background for the scripts to connect to the local LLM.
- For more models and advanced usage, see the [Ollama documentation](https://ollama.com/library).

---
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

Local-LLM is an open source project that makes it easy to run Large Language Models (LLMs) entirely offline on your own machine using [Ollama](https://ollama.com/). This project is designed for privacy, speed, and full local control—no cloud required!

This project is licensed under the [MIT License](./LICENSE).