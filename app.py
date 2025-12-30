import streamlit as st
import os
import sys
import io

# 1. Kodlaşdırma üçün ən sərt tənzimləmə
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="ATA.ai", layout="centered")

if os.path.exists("logo.png"):
    st.image("logo.png", width=150)

st.title("Azərbaycan Tarixi Akademiyası")
st.markdown("---")

# API AÇARI
gemini_key = "BURAYA_GEMINI_API_AÇARINI_YAZ"

@st.cache_resource
def sistemi_yukle():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Bazanı yükləyirik
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_key)
    return vector_db, llm

if gemini_key == "AIzaSyD__8kh00rQvmO7ezNjgGeO8ZZY-qMOYb8":
    st.warning("Lütfən, koda Gemini API açarınızı daxil edin.")
else:
    try:
        db, llm = sistemi_yukle()
        
        # Sual daxil etmə (UTF-8 dəstəyi ilə)
        query = st.text_input("Tariximiz haqqında nə öyrənmək istəyirsiniz?")

        if query:
            with st.spinner("Tarixi sənədlər araşdırılır..."):
                # Sənədlərdə axtarış
                docs = db.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Promptu birbaşa string kimi yox, f-string ilə UTF-8 olaraq ötürürük
                full_prompt = f"Məlumat: {context}\n\nSual: {query}\n\nCavabı yalnız Azərbaycan dilində ver."
                
                # Modeldən cavabı alırıq
                response = llm.invoke(full_prompt)
                
                st.markdown("### Cavab:")
                # Cavabı göstərərkən xətanın qarşısını almaq üçün .content istifadə edirik
                st.success(response.content)
                
    except Exception as e:
        # Xəta mesajını UTF-8-ə çevirib göstəririk
        error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
        st.error(f"Texniki xəta: {error_msg}")