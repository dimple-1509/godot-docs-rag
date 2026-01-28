import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter as rcts  #rcts is actually a class

DOCS_PATH = "godot-docs"
INDEX_PATH = "godot_index"

def load_godot_docs():
    all_text = []

    EXCLUDED_DIRS = {
        ".git", ".github", "_static", "locale"
    }

    EXCLUDED_FILES = {
        ".gitattributes","Makefile", "robots.txt", ".gitignore", ".editorconfig", ".mailmap",".lycheeignore",".pre-commit-config.yaml",
        ".readthedocs.yml","404.rst","AUTHORS.md","conf.py","indes.rst","LICENSE.txt","make.bat","Makefile","pyproject.toml","README.md","requirements.txt"
    }

    for root, dirs, files in os.walk(DOCS_PATH):
        # remove unwanted dirs in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if file in EXCLUDED_FILES:
                continue

            if file.endswith((".rst", ".md")):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        all_text.append(f.read())
                except Exception as e:
                    print(f"Skipped {file_path}: {e}")
    print("loaded godot docs")
    return "\n".join(all_text)


def split_into_chunks():

    txt_data = load_godot_docs()

    text_splitter = rcts(
        separators = "\n",
        chunk_size = 300,
        chunk_overlap = 50 #----
    )# IT CREATES A OBJECT WITH SEVERAL ATTRIBUTES

    chunks = text_splitter.split_text(txt_data)  #USING THE OBJECT CREATED WE CALL THE METHOD

    clean_chunks = [
    chunk.replace("\r\n", " ").replace("\n", " ").replace('"',"").strip() + "..."
    for chunk in chunks
]
    print("splitted into chunks")
    return clean_chunks


clean_chunks = split_into_chunks()

def chunks_into_embeddings(clean_chunks):
    documents = [Document(page_content=chunk) for chunk in clean_chunks]

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("initialised embedding")


    # vectorstore = FAISS.from_documents(documents,embeddings)
    #IT IS ACTUALLY THE DATABASE WHICH STORES BOTH THE DOCUMENT CONTENT AND THE VECTOR LIST CORRESPONDING TO IT


    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"Index built with {len(documents)} chunks")

chunks_into_embeddings(clean_chunks)