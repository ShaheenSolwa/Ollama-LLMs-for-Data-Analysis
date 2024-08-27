import ollama
import pandas as pd


def read_file(file_path):
    """Read the contents of a CSV or Excel file and return them as a string."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

    # Convert the DataFrame to a string
    return df.to_string(index=False)


def main(file_path):
    # Read the file content
    file_content = read_file(file_path)

    # Prepare the message content
    message_content = (f"Here is the data from the file:\n\n{file_content}\n\nPlease analyze it and give me statistics for each "
                       f"column, totals for each column and unique values for each column")

    # Initialize the chat stream
    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': message_content}],
        stream=True
    )

    # Print the streamed response
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


# Replace 'your_file.csv' with the path to your CSV or XLSX file
if __name__ == "__main__":
    file_path = r"C:\Users\FTS SERVER\Downloads\articles to summarize\tips.csv"  # or 'your_file.xlsx'
    main(file_path)
