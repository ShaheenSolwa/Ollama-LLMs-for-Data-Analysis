import streamlit as st
import fitz  # PyMuPDF
import ollama

def read_pdf_file(file):
    """Read the contents of a PDF file and return them as a string."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def chat_with_model(messages):
    """Interact with the Ollama chat model."""
    try:
        stream = ollama.chat(
            model='llama3.1',
            messages=messages,
            stream=True
        )
        response_content = ''
        for chunk in stream:
            response_content += chunk.get('message', {}).get('content', '')
        return response_content
    except Exception as e:
        st.error(f"Error communicating with the model: {e}")
        return ""

def main():
    st.title("Chat with Data Analysis")

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'file_content' not in st.session_state:
        st.session_state.file_content = None

    # File upload widget
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'], accept_multiple_files=False)

    if uploaded_file:
        # Read and store file content
        st.session_state.file_content = read_pdf_file(uploaded_file)
        if st.session_state.file_content:
            st.success("File uploaded and read successfully.")
        else:
            st.session_state.file_content = None

    # Display chat history
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Model:** {msg['content']}")

    # User input for custom messages
    if st.session_state.file_content is not None:
        user_input = st.text_input("Type your question here:")

        if st.button("Send"):
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({'role': 'user', 'content': user_input})

                # Combine file content with the user's message
                combined_message = f"Here is the data from the file:\n\n{st.session_state.file_content}\n\nUser question: {user_input}"
                st.session_state.messages.append({'role': 'user', 'content': combined_message})

                # Get response from the chat model
                with st.spinner("Generating response..."):
                    response = chat_with_model(st.session_state.messages)
                    if response:
                        st.session_state.messages.append({'role': 'assistant', 'content': response})
                    else:
                        st.error("Failed to get a response from the model.")

if __name__ == "__main__":
    main()
