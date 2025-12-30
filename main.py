import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Bütün PDF-ləri data qovluğundan yüklə
print("[1/3] ATA.ai: Sənədlər qovluğu oxunur...")
loader = DirectoryLoader('data/', glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
print(f"Cəmi {len(docs)} səhifə tapıldı.")

# 2. Mətnləri hissələrə böl
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
texts = text_splitter.split_documents(docs)
print(f"Cəmi {len(texts)} mətn hissəsi yaradıldı.")

# 3. Yaddaş bazasını hissə-hissə qur
print("[2/3] Yaddaş bazası qurulur (RAM qorunur)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# İlk 50 hissə ilə bazanı başlat
vector_db = FAISS.from_documents(texts[:50], embeddings)

# Qalanlarını 50-50 əlavə et ki, kompüter donmasın
for i in range(50, len(texts), 50):
    batch = texts[i:i+50]
    vector_db.add_documents(batch)
    if i % 250 == 0:
        print(f"Tərəqqi: {i}/{len(texts)} hissə emal olundu...")

# 4. Yaddaşı saxla
vector_db.save_local("faiss_index")
print("[3/3] SİSTEM HAZIRDIR! Artıq sual-cavab edə bilərsiniz.")