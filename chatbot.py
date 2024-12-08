# chatbot.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

load_dotenv()

class Chatbot:
    def __init__(self, model_name="openai"):
        self.model_name = model_name
        self.chat_history = []
        self._init_main_models()
        
        # AIML API setup
        self.aiml_client = OpenAI(
            base_url="https://api.aimlapi.com/v1",
            api_key="8bbffe6cf7134a24a8a17f2b002bbe3b"
        )

    def _init_main_models(self):
        self.openai_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.7
        )
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model = genai.GenerativeModel("gemini-pro")
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def _generate_aiml_response(self, user_input: str) -> str:
        try:
            messages = [
                {"role": "system", "content": "Tu es un assistant IA sympathique et serviable."},
                *[{"role": msg["role"], "content": msg["content"]} for msg in self.chat_history],
                {"role": "user", "content": user_input}
            ]

            completion = self.aiml_client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=messages,
                temperature=0.7,
                max_tokens=256
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"AIML API Error: {str(e)}")
            return f"Désolé, une erreur s'est produite avec l'API AIML: {str(e)}"

    def update_model(self, new_model_name):
        self.model_name = new_model_name
        self.clear_context()

    def get_response(self, user_input: str) -> str:
        try:
            user_input_lower = user_input.lower()
            if "aide" in user_input_lower or "help" in user_input_lower:
                return self._get_help_message()
            
            if any(word in user_input_lower for word in ["bonjour", "salut", "hello", "hi"]):
                return "Bonjour! Comment puis-je vous aider?"
            
            if any(word in user_input_lower for word in ["merci", "thanks"]):
                return "Je vous en prie!"

            if self.model_name == "openai":
                messages = [{"role": "system", "content": "Tu es un assistant IA sympathique et serviable."}]
                messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in self.chat_history])
                messages.append({"role": "user", "content": user_input})
                response = self.openai_model.invoke(messages).content
                
            elif self.model_name == "gemini":
                response = self.gemini_model.generate_content(user_input).text
                
            elif self.model_name == "groq":
                messages = [{"role": "system", "content": "Tu es un assistant IA sympathique et serviable."}]
                messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in self.chat_history])
                messages.append({"role": "user", "content": user_input})
                completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model="mixtral-8x7b-32768",
                    temperature=0.7
                )
                response = completion.choices[0].message.content
                
            elif self.model_name == "aiml":
                response = self._generate_aiml_response(user_input)
            
            else:
                raise ValueError(f"Modèle non supporté: {self.model_name}")

            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"Désolé, une erreur s'est produite: {str(e)}"

    def _get_help_message(self) -> str:
        return """Je suis là pour vous aider! Voici mes capacités:
               1. Réponses aux questions générales
               2. Assistance pour diverses tâches
               3. Modèles disponibles: OpenAI, Groq, Gemini, AIML"""

    def clear_context(self):
        self.chat_history = []