import tkinter as tk
from tkinter import scrolledtext, messagebox
from huggingface_hub import login
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Function to check and perform login
def perform_login():
    login_file = "hf_login_done.txt"
    if not os.path.exists(login_file):
        os.environ["HF_KEY"] = "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"
        login(token=os.environ.get('HF_KEY'), add_to_git_credential=False)
        with open(login_file, "w") as f:
            f.write("Login done")

# Perform login if not done already
perform_login()

# Load the tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def prepare_prompt(natural_language_query, schema):
    schema_description = "\n".join(
        [f"Table: {table}\nColumns: {', '.join(columns)}" for table, columns in schema.items()]
    )
    prompt = (f"Translate the following natural language query into an SQL query "
              f"based on the schema provided:\n\n"
              f"Schema:\n{schema_description}\n\n"
              f"Query:\n{natural_language_query}")
    return prompt

def generate_sql_query(natural_language_query, schema):
    prompt = prepare_prompt(natural_language_query, schema)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

def parse_schema(schema_str):
    schema = {}
    tables = schema_str.split('|')
    for table in tables:
        if ':' in table:
            table_name, cols = table.split(':')
            cols = [col.strip() for col in cols.split(',')]
            schema[table_name.strip()] = cols
    return schema

def on_generate_click():
    schema_input = schema_text.get("1.0", tk.END).strip()
    natural_language_query = query_entry.get().strip()
    if natural_language_query and schema_input:
        schema = parse_schema(schema_input)
        if schema:
            try:
                sql_query = generate_sql_query(natural_language_query, schema)
                result_text.config(state=tk.NORMAL)
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, sql_query)
                result_text.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            messagebox.showerror("Error", "Please enter a valid schema.")
    else:
        messagebox.showerror("Error", "Please enter both schema and query.")

# Create the main window
root = tk.Tk()
root.title("SQL Query Generator")

# Layout
tk.Label(root, text="Define Your Database Schema").pack(pady=5)
schema_text = scrolledtext.ScrolledText(root, width=60, height=10)
schema_text.pack(pady=5)

tk.Label(root, text="Enter your query:").pack(pady=5)
query_entry = tk.Entry(root, width=60)
query_entry.pack(pady=5)

generate_button = tk.Button(root, text="Generate SQL Query", command=on_generate_click)
generate_button.pack(pady=10)

tk.Label(root, text="Generated SQL Query:").pack(pady=5)
result_text = scrolledtext.ScrolledText(root, width=60, height=10, state=tk.DISABLED)
result_text.pack(pady=5)

# Run the application
root.mainloop()
