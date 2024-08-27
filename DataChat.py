import os, logging, sys, warnings

warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from huggingface_hub import login

os.environ["HF_KEY"] = "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"
login(token=os.environ.get('HF_KEY'),add_to_git_credential=False)

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_dir=r"C:\Users\FTS SERVER\Downloads\articles to summarize",
                                  required_exts=[".csv", ".xlsx"]).load_data()



from langchain.embeddings.huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents, embed_model = embed_model)

#from llama_index.core.prompts.prompts import PromptTemplate
from llama_index.core import PromptTemplate

system_prompt = """<|SYSTEM|>
You are an expert data analyst that can compute complex data. 
You need to answer the questions using only the context provided.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

LLM_MODEL_NAME = "tiiuae/falcon-7b"
    #"meta-llama/Llama-2-7b-chat-hf"

import torch
from llama_index.llms.huggingface import HuggingFaceLLM

# To import models from HuggingFace directly
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=LLM_MODEL_NAME,
    model_name=LLM_MODEL_NAME,
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    #model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

from llama_index.core import Settings

Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 1024
#Settings.chunk_overlap = 256

query_engine = index.as_query_engine(llm=llm, similarity_top_k=1)

done = False
while not done:
  print("*"*30)
  question = input("Enter your question: ")
  response = query_engine.query(question)
  print(response)
  done = input("End the chat? (y/n): ") == "y"
