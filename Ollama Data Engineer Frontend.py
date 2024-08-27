import streamlit as st
import pandas as pd
import ollama

def read_file(file):
    """Read the contents of a CSV or Excel file and return them as a string."""
    if file.type == 'text/csv':
        df = pd.read_csv(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

    # Convert the DataFrame to a string
    return df.to_string(index=False)

def chat_with_model(messages):
    """Interact with the Ollama chat model."""
    # Initialize the chat stream
    stream = ollama.chat(
        model='llama3.1',
        messages=messages,
        stream=True
    )

    # Collect the streamed response
    response_content = ''
    for chunk in stream:
        response_content += chunk['message']['content']

    return response_content

def main():
    st.title("Chat with Data Analysis")

    # Session state for chat history and file content
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'file_content' not in st.session_state:
        st.session_state.file_content = None

    # File upload widget
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            # Read the file content
            st.session_state.file_content = read_file(uploaded_file)

        except ValueError as e:
            st.error(e)

    # Display chat history
    if st.session_state.messages:
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Model:** {msg['content']}")

    # User input for custom messages
    if st.session_state.file_content is not None:
        user_input = st.text_input("Type your question here:", key="user_input")

        if st.button("Send"):
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({'role': 'user', 'content': user_input})

                # Include the file content in the message
                message_content = f"Here is the data from the file:\n\n{st.session_state.file_content}\n\nUser question: {user_input}"
                st.session_state.messages.append({'role': 'user', 'content': message_content})

                # Get response from the chat model
                with st.spinner("Generating response..."):
                    response = chat_with_model(st.session_state.messages)

                # Add the model response to chat history
                st.session_state.messages.append({'role': 'assistant', 'content': response})

                # Clear the input field by using st.session_state directly
                st.session_state.user_input = ""

if __name__ == "__main__":
    main()
