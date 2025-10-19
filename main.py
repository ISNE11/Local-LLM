import sys
import os
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal
from llm_chat_gui import ChatWindow
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local LLM Main")
        self.resize(800, 700)
        self.layout = QVBoxLayout()

        # Top controls
        controls_layout = QHBoxLayout()
        self.textbook_checkbox = QCheckBox("Use Textbook")
        self.textbook_checkbox.setChecked(True)
        self.textbook_checkbox.stateChanged.connect(self.on_textbook_toggle)
        controls_layout.addWidget(self.textbook_checkbox)

        self.open_ended_checkbox = QCheckBox("Open Ended Question")
        self.open_ended_checkbox.setChecked(True)
        self.open_ended_checkbox.stateChanged.connect(self.on_question_type_toggle)
        controls_layout.addWidget(self.open_ended_checkbox)

        self.multiple_choice_checkbox = QCheckBox("Multiple Choice Question")
        self.multiple_choice_checkbox.setChecked(False)
        self.multiple_choice_checkbox.stateChanged.connect(self.on_question_type_toggle)
        controls_layout.addWidget(self.multiple_choice_checkbox)

        from PyQt5.QtWidgets import QPushButton
        self.stop_button = QPushButton("Stop AI")
        self.stop_button.clicked.connect(self.stop_ai)
        controls_layout.addWidget(self.stop_button)

        self.layout.addLayout(controls_layout)

        # Chat window
        self.chat_window = ChatWindow()
        self.layout.addWidget(self.chat_window)

        self.setLayout(self.layout)

        # State
        self.conversation_history = []
        self.use_textbook = True
        self.question_type = "open_ended"  # or "multiple_choice"
        # Textbook/Vector DB/LLM setup
        self.TEXTBOOK_PATH = "textbook.txt"
        self.VECTOR_DB_DIR = "./db"
        self.EMBED_MODEL = "nomic-embed-text"
        self.LLM_MODEL = "deepseek-r1:8b"
        self.llm = OllamaLLM(model=self.LLM_MODEL)
        self.db = None
        self._init_textbook_db()
        # Connect chat send
        self.chat_window.send_message = self.send_message_with_context

    def stop_ai(self):
        if self.chat_window.llm_thread is not None and self.chat_window._streaming:
            self.chat_window.llm_thread.stop()
            self.chat_window._streaming = False
            self.chat_window.append_message("[AI stopped by user]", user=False)
        self.LLM_MODEL = "deepseek-r1:8b"
        self.llm = OllamaLLM(model=self.LLM_MODEL)
        self.db = None
        self._init_textbook_db()
        # Connect chat send
        self.chat_window.send_message = self.send_message_with_context

    def on_textbook_toggle(self, state):
        self.use_textbook = state == 2

    def on_question_type_toggle(self, state):
        sender = self.sender()
        if sender == self.open_ended_checkbox and state == 2:
            self.multiple_choice_checkbox.setChecked(False)
            self.question_type = "open_ended"
        elif sender == self.multiple_choice_checkbox and state == 2:
            self.open_ended_checkbox.setChecked(False)
            self.question_type = "multiple_choice"
        elif state == 0:
            if not self.open_ended_checkbox.isChecked() and not self.multiple_choice_checkbox.isChecked():
                self.open_ended_checkbox.setChecked(True)
                self.question_type = "open_ended"

    def _init_textbook_db(self):
        if not os.path.exists(self.TEXTBOOK_PATH):
            self.db = None
            return
        loader = TextLoader(self.TEXTBOOK_PATH, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        emb = OllamaEmbeddings(model=self.EMBED_MODEL)
        if os.path.exists(self.VECTOR_DB_DIR) and os.listdir(self.VECTOR_DB_DIR):
            self.db = Chroma(persist_directory=self.VECTOR_DB_DIR, embedding_function=emb)
        else:
            self.db = Chroma.from_documents(texts, emb, persist_directory=self.VECTOR_DB_DIR)

    def send_message_with_context(self):
        user_text = self.chat_window.input_box.text().strip()
        if not user_text or self.chat_window._streaming:
            return
        self.chat_window.append_message(user_text, user=True)
        self.chat_window.input_box.clear()
        self.chat_window.append_message("", user=False, streaming=True)
        # Build context and prompt
        context = ""
        if self.use_textbook and self.db is not None:
            docs = self.db.similarity_search(user_text, k=5)
            seen = set()
            unique_texts = [d.page_content for d in docs if not (d.page_content in seen or seen.add(d.page_content))]
            context = "\n".join(unique_texts)
        prompt = self.build_prompt(user_text, context)
        # Start LLM thread
        if self.chat_window.llm_thread is not None:
            self.chat_window.llm_thread.stop()
            self.chat_window.llm_thread.wait()
        self.chat_window.llm_thread = LLMWorkerWithPrompt(prompt, self.llm, self)
        self.chat_window.llm_thread.chunk_signal.connect(self.chat_window.update_stream)
        self.chat_window.llm_thread.done_signal.connect(self.chat_window.finish_stream)
        self.chat_window._streaming = True
        self.chat_window.llm_thread.start()

    def build_prompt(self, user_query, context):
        if self.question_type == "multiple_choice":
            return f"""
You are an expert at answering multiple-choice questions based on the provided textbook content.

Textbook content:
{context}

Question: {user_query}

Instructions:
1. First, give the correct answer concisely (e.g., "Answer: B").
2. Then provide a step-by-step explanation below your answer.
"""
        else:
            # With history, open-ended
            self.conversation_history.append(f"You: {user_query}")
            history = "\n".join(self.conversation_history[-5:])
            prompt = f"""
Conversation history:
{history}  # Only keep last 5 turns for speed

Use the following textbook content as your primary reference.
If necessary, reason with general knowledge but prioritize textbook facts.
Avoid repeating information that was already explained earlier.

Textbook content:
{context}

Question: {user_query}
"""
            return prompt

class LLMWorkerWithPrompt(QThread):
    chunk_signal = pyqtSignal(str)
    done_signal = pyqtSignal()
    def __init__(self, prompt, llm, main_window):
        super().__init__()
        self.prompt = prompt
        self.llm = llm
        self.main_window = main_window
        self._running = True
    def run(self):
        response = ""
        for chunk in self.llm.stream(self.prompt):
            if not self._running:
                break
            response += chunk
            self.chunk_signal.emit(response)
        self.done_signal.emit()
        # Save bot response to history if open-ended
        if self.main_window.question_type == "open_ended":
            self.main_window.conversation_history.append(f"DeepSeek: {response}")
    def stop(self):
        self._running = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())