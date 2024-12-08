# app_streamlit.py
import streamlit as st
from chatbot import Chatbot

def initialize_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = Chatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "openai"

def main():
    st.title("🤖 Assistant IA Multi-Modèles")
    
    initialize_session_state()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        model_options = {
            'openai': 'OpenAI GPT-3.5',
            'groq': 'Groq (Mixtral)',
            'gemini': 'Google Gemini Pro',
            'aiml': 'AIML API'
        }
        
        selected_model = st.selectbox(
            "Choisir le modèle d'IA",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        if selected_model != st.session_state.current_model:
            with st.spinner("Changement de modèle..."):
                st.session_state.chatbot.update_model(selected_model)
                st.session_state.current_model = selected_model
                st.session_state.messages = []
                st.rerun()
    
    with col2:
        if st.button("🗑️ Effacer"):
            st.session_state.chatbot.clear_context()
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Votre message...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                response = st.session_state.chatbot.get_response(user_input)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()