import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter as rcts  #rcts is actually a class
from langchain_core.prompts import ChatPromptTemplate


def load_godot_docs(base_path="godot-docs"):
    all_text = []

    EXCLUDED_DIRS = {
        ".git", ".github", "_static", "locale"
    }

    EXCLUDED_FILES = {
        ".gitattributes","Makefile", "robots.txt", ".gitignore", ".editorconfig", ".mailmap",".lycheeignore",".pre-commit-config.yaml",
        ".readthedocs.yml","404.rst","AUTHORS.md","conf.py","indes.rst","LICENSE.txt","make.bat","Makefile","pyproject.toml","README.md","requirements.txt"
    }

    for root, dirs, files in os.walk(base_path):
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
    document = [Document(page_content=chunk) for chunk in clean_chunks]

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("initialised embedding")


    # vectorstore = FAISS.from_documents(document,embeddings)
    #IT IS ACTUALLY THE DATABASE WHICH STORES BOTH THE DOCUMENT CONTENT AND THE VECTOR LIST CORRESPONDING TO IT


    BATCH_SIZE = 100

    first_batch = document[:BATCH_SIZE]
    vectorstore = FAISS.from_documents(first_batch, embeddings)
    print("vectorstore")
    

    for i in range(BATCH_SIZE, len(document), BATCH_SIZE):
        batch = document[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Embedded {i + len(batch)} / {len(document)}")

    vectorstore.save_local("godot_faiss")

    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature = 0
    )

    print("stored chunks with their embeddings")
    question = "How to use animation tool in godot engine"

# --- similarity search with scores ---
    docs_and_scores = vectorstore.similarity_search_with_score(
    question, k=2
  )

#     SIMILARITY_THRESHOLD = 0.00  # you can tune this

    # filtered_docs = [
    #  doc for doc, score in docs_and_scores
    #  if score < SIMILARITY_THRESHOLD
    # ]

 # --- no relevant docs found ---
#     if not filtered_docs:
#       print("I don't know. This information is not present in my documents.")
#       return

# --- build context ---
    context_text = "\n".join(item[0].page_content for item in docs_and_scores)

    if len(context_text.strip()) < 50:
     print("I don't know. This information is not present in my documents.")
     return


# --- strict system prompt ---
    SYSTEM_PROMPT = """
You are a retrieval-based assistant.

RULES:
- Answer ONLY using the provided context.
- Search through the context provided properly and carefully
- If the answer is not present in the context, say:
  "I don't know based on the provided documents."
- Do NOT use prior knowledge.
- Do NOT guess.
"""

    prompt = ChatPromptTemplate.from_messages([
     ("system", SYSTEM_PROMPT),
     ("human", "Context:\n{context}\n\nQuestion:\n{question}")
   ])

    # Build the prompt Runnable
    prompt_runnable = prompt

# Combine prompt Runnable with LLM 
    runnable = prompt_runnable | llm

# Execute with invoke()
    print("calling the model for answer ")
    answer = runnable.invoke({
      "context": context_text,
      "question": question
    })


    print("\nFinal answer____\n")
    print(answer.content)


chunks_into_embeddings(clean_chunks)


