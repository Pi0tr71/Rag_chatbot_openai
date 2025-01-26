import streamlit as st
import os
from retrieval import retrieve_docs, build_prompt
from models.openai_llm import openai_chat_completion

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def chat_with_user(user_question: str):


    context_chunks = retrieve_docs(user_question, top_k=3)
    system_prompt = (
        """You're an AI assistant that answers questions based solely on the provided text fragments.
At the end of your response, include the sources in parentheses, e.g., (Source: file.txt), but only if the information comes from the provided fragments.
If the information is insufficient, let the user know that you're not sure or you don't know. Do not include a source if the information is not available in the provided text fragments.
Do not use any external knowledge or information beyond the provided text fragments."""
    )
    prompt = build_prompt(system_prompt, context_chunks, user_question)
    answer = openai_chat_completion(prompt)
    return answer


def main():
    st.title("Chatbot powered by Retrieval-Augmented Generation")
    st.write("Ask your question, and the chatbot will answer based solely on the provided text fragments.")

    # Pole tekstowe do pytania użytkownika
    user_question = st.text_input("Enter your question:", placeholder="Type your question here...")

    if st.button("Ask the Chatbot"):
        if user_question.strip():
            with st.spinner("Fetching the answer..."):
                try:
                    answer = chat_with_user(user_question)
                    st.success("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question before clicking 'Ask the Chatbot'.")

if __name__ == "__main__":
    main()