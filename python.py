
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter as rcts  #rcts is actually a class
from langchain_core.prompts import ChatPromptTemplate


def split_into_chunks():
    file_path = 'info.txt'

    with open(file_path,'r',newline='',encoding='utf-8') as file:
        txt_data = file.read()

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

    return clean_chunks


clean_chunks = split_into_chunks()



def chunks_into_embeddings(clean_chunks):
    document = [Document(page_content=chunk) for chunk in clean_chunks]

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(document,embeddings)
    #IT IS ACTUALLY THE DATABASE WHICH STORES BOTH THE DOCUMENT CONTENT AND THE VECTOR LIST CORRESPONDING TO IT

    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature = 0
    )


    question = "Is godot free"

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
    answer = runnable.invoke({
      "context": context_text,
      "question": question
    })


    print("\nFinal answer____\n")
    print(answer.content)


chunks_into_embeddings(clean_chunks)


