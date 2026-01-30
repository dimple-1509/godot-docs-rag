import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter as rcts  #rcts is actually a class

DOCS_PATH = "godot-docs"
INDEX_PATH = "godot_index"

def load_godot_docs():
    documents = []

    EXCLUDED_DIRS = {
        ".git", ".github", "_static", "locale"
    }

    EXCLUDED_FILES = {
        ".gitattributes","Makefile", "robots.txt", ".gitignore", ".editorconfig", ".mailmap",".lycheeignore",".pre-commit-config.yaml",
        ".readthedocs.yml","404.rst","AUTHORS.md","conf.py","indes.rst","LICENSE.txt","make.bat","Makefile","pyproject.toml","README.md","requirements.txt"
    }

    text_splitter = rcts(
        separators = "\n",
        chunk_size = 300,
        chunk_overlap = 50 #----
    )# IT CREATES A OBJECT WITH SEVERAL ATTRIBUTES

    for root, dirs, files in os.walk(DOCS_PATH):
        # remove unwanted dirs in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if file in EXCLUDED_FILES:
                continue

            if not file.endswith((".rst", ".md")):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = text_splitter.split_text(text)

                for chunk in chunks:
                    documents.append(Document(
                        page_content = chunk,
                        metadata = {
                            "source" : file_path
                        }
                    ))

            except Exception as e:
                 print(f"Skipped {file_path}: {e}")

    print("Loaded and Splitted Godot docs data")
    return documents


def build_index():

    documents = load_godot_docs()


    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("initialised embedding","documents size: ",len(documents))

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"Index built with {len(documents)} chunks")



if __name__ == "__main__":
    build_index()