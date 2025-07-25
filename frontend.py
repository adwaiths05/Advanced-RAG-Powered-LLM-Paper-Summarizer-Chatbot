import streamlit as st
import requests

st.title("LLM Papers Chatbot")
st.write("Query the RAG-powered chatbot for insights from 100 LLM papers.")

query = st.text_input("Enter your query:", placeholder="What is LLM research?")

if st.button("Submit"):
    if query:
        try:
            response = requests.get("http://localhost:8000/query", params={"query": query}, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.success("Response:")
            st.write(result["response"])
        except requests.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please enter a query.")
