import tkinter as tk
from tkinter import scrolledtext
import os
import logging
import sys
import warnings
import threading
from huggingface_hub import login
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

# Initialize logging
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Authenticate HuggingFace
os.environ["HF_KEY"] = "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"
login(token=os.environ.get('HF_KEY'), add_to_git_credential=False)

# Load documents (should be done once at the start)
documents = SimpleDirectoryReader(
    input_dir=r"C:\Users\FTS SERVER\Downloads\articles to summarize",
    required_exts=[".pdf", ".docx", ".txt"]
).load_data()

# Set up embedding and LLM models
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

system_prompt = """# You are an AI-enabled admin assistant.
Your goal is to answer questions accurately using only the context provided.
"""
query_wrapper_prompt = PromptTemplate("{query_str}")
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=LLM_MODEL_NAME,
    model_name=LLM_MODEL_NAME,
    device_map="auto"
)

Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 1024

query_engine = index.as_query_engine(llm=llm, similarity_top_k=1)

# Set up GUI
class ChatFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat Interface")

        self.chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap='word')
        self.chat_area.pack(padx=10, pady=10, fill='both', expand=True)

        self.entry = tk.Entry(root)
        self.entry.bind("<Return>", self.send_message)
        self.entry.pack(padx=10, pady=10, fill='x')

    def append_to_chat_area(self, text):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, text)
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def send_message(self, event=None):
        user_input = self.entry.get()
        if user_input:
            self.append_to_chat_area(f"You: {user_input}\n")
            self.entry.delete(0, tk.END)

            # Run the query in a separate thread
            threading.Thread(target=self.process_query, args=(user_input,), daemon=True).start()

    def process_query(self, user_input):
        try:
            # Run the query and get the response
            response = query_engine.query(user_input)
            # Update the chat area with the response
            self.root.after(0, self.append_to_chat_area, f"AI: {response}\n")
        except Exception as e:
            self.root.after(0, self.append_to_chat_area, f"Error: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatFrontend(root)
    root.mainloop()