from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

INDEX_PATH = "godot_index"


def model(question,INDEX_PATH = INDEX_PATH):
    embeddingModel = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorDB = FAISS.load_local(
        INDEX_PATH, embeddingModel, allow_dangerous_deserialization = True
    )

    LLM = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature = 0
    )


# --- similarity search with scores ---
    docs_and_scores = vectorDB.similarity_search_with_score(question,k=3)

#  --- filtering based on similarity threshold ---
    SIMILARITY_THRESHOLD = 0.4

    filtered_docs = [
        doc for doc,score in docs_and_scores if score < SIMILARITY_THRESHOLD
    ]

# -- if no relevant docs found --- 
    if not filtered_docs:
        print(" I don't know. This information is not present in the provided documents. ")
        return

# -- prepare context text ---
    context_text = "\n".join(doc.page_content for doc in filtered_docs)


# --- prompt template ---
    SYSTEM_PROMPT = """
    You are a retrieval - based assistant.

    RULES:
    - Answer ONLY using the PROVIDED CONTEXT.
    - Search through the context provided properly and carefully.
    - If the answer is not present in the context, say:
      "I don't know based on the provided documents. "
    - Do NOT use prior knowledge.
    - Do NOT guess.
"""

# --- build prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system",SYSTEM_PROMPT),
        ("human","Context:\n{context}\n\nQuestion:\n{question}")
    ])

    prompt_filled = prompt.format_prompt(
        context = context_text,
        question = question
    )

    messages = prompt_filled.to_messages()


# --- Get the answer from the LLM ---
    answer = LLM.generate([messages])  

    print("\n Final answer____\n")

    print(answer.generations[0][0].text)



if __name__ == "__main__":
    user_question = input("Please enter your question about Godot Engine: ")
    model(user_question)





