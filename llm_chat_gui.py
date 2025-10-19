import sys
import markdown
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from langchain_ollama import OllamaLLM

LLM_MODEL = "deepseek-r1:8b"
llm = OllamaLLM(model=LLM_MODEL)

class LLMWorker(QThread):
    chunk_signal = pyqtSignal(str)
    done_signal = pyqtSignal()
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        self._running = True
    def run(self):
        response = ""
        for chunk in llm.stream(self.prompt):
            if not self._running:
                break
            response += chunk
            self.chunk_signal.emit(response)
        self.done_signal.emit()
    def stop(self):
        self._running = False

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local LLM Chat")
        self.resize(700, 600)
        self.layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message and press Enter...")
        self.send_button = QPushButton("Send")
        self.layout.addWidget(self.chat_display)
        self.layout.addWidget(self.input_box)
        self.layout.addWidget(self.send_button)
        self.setLayout(self.layout)
        self.input_box.returnPressed.connect(self.send_message)
        self.send_button.clicked.connect(self.send_message)
        self.llm_thread = None
        self._streaming = False
        self._last_bot_start = None
        self._last_bot_end = None

    def send_message(self):
        user_text = self.input_box.text().strip()
        if not user_text or self._streaming:
            return
        self.append_message(user_text, user=True)
        self.input_box.clear()
        self.append_message("", user=False, streaming=True)  # Placeholder for streaming
        if self.llm_thread is not None:
            self.llm_thread.stop()
            self.llm_thread.wait()
        self.llm_thread = LLMWorker(user_text)
        self.llm_thread.chunk_signal.connect(self.update_stream)
        self.llm_thread.done_signal.connect(self.finish_stream)
        self._streaming = True
        self.llm_thread.start()

    def append_message(self, message, user=False, streaming=False):
        scrollbar = self.chat_display.verticalScrollBar()
        # Check if user is at the bottom before update
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 2  # allow small margin
        html = markdown.markdown(message, extensions=['fenced_code', 'tables'])
        if user:
            html = f'<div style="color:#007acc;"><b>You:</b> {html}</div>'
        else:
            html = f'<div style="color:#222;"><b>AI:</b> {html}</div>'
        self.chat_display.moveCursor(QTextCursor.End)
        start = self.chat_display.textCursor().position()
        self.chat_display.insertHtml(html + "<br><br>")
        end = self.chat_display.textCursor().position()
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        if streaming:
            self._last_bot_start = start
            self._last_bot_end = end

    def update_stream(self, text):
        if self._last_bot_start is not None and self._last_bot_end is not None:
            scrollbar = self.chat_display.verticalScrollBar()
            # Check if user is at the bottom before update
            at_bottom = scrollbar.value() >= scrollbar.maximum() - 2  # allow small margin
            cursor = self.chat_display.textCursor()
            cursor.setPosition(self._last_bot_start)
            cursor.setPosition(self._last_bot_end, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
            html = f'<div style="color:#222;"><b>AI:</b> {html}</div>'
            cursor.insertHtml(html + "<br><br>")
            self._last_bot_end = cursor.position()
            if at_bottom:
                scrollbar.setValue(scrollbar.maximum())

    def finish_stream(self):
        self._streaming = False
        self._last_bot_start = None
        self._last_bot_end = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
