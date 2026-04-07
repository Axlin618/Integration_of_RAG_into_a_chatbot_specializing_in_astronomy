import streamlit as st
import pickle
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


genai.configure(api_key="API Key") # Aquí debes poner poner tu API Key de Gemini
modelo_IA = genai.GenerativeModel("models/gemini-2.5-flash")

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

vector_db = FAISS.load_local(
    "faiss_astro_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def buscar_contexto(pregunta, k=5):
    return vector_db.similarity_search(pregunta, k=k)

def construir_prompt(docs, pregunta, historial):
    contexto = "\n".join([doc.page_content for doc in docs])
    historial_texto = "\n".join([f"{r}: {m}" for r, m in historial])

    return f"""
Eres un experto en astronomía.

Historial:
{historial_texto}

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma clara y completa.
"""

st.set_page_config(page_title="Asistente Astronómico", layout="wide")
DB_PATH = Path("chats.pkl")

if DB_PATH.exists():
    with open(DB_PATH, "rb") as f:
        chats = pickle.load(f)
else:
    chats = {}

st.sidebar.title("Historial de chats")

if st.sidebar.button("Nuevo chat"):
    nombre = f"Chat {len(chats)+1}"
    chats[nombre] = []
    with open(DB_PATH, "wb") as f:
        pickle.dump(chats, f)
    st.rerun()

chat_actual = st.sidebar.selectbox(
    "Selecciona chat",
    list(chats.keys()) if chats else ["Chat 1"]
)

if chat_actual not in chats:
    chats[chat_actual] = []

if st.sidebar.button("Borrar chat"):
    chats.pop(chat_actual)
    with open(DB_PATH, "wb") as f:
        pickle.dump(chats, f)
    st.rerun()

st.title("✨ Chatbot de Astronomía ✨")

messages = chats[chat_actual]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pregunta cualquier cosa de astronomía =)"):

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta_final = ""

        with st.spinner("Pensando..."):

            docs = buscar_contexto(prompt)
            historial = [(m["role"], m["content"]) for m in messages]

            prompt_final = construir_prompt(docs, prompt, historial)
            respuesta = modelo_IA.generate_content(prompt_final).text

        # efecto escritura 
        for char in respuesta:
            respuesta_final += char
            placeholder.markdown(respuesta_final)

        messages.append({"role": "assistant", "content": respuesta})

        chats[chat_actual] = messages
        with open(DB_PATH, "wb") as f:
            pickle.dump(chats, f)