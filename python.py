from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

INDEX_PATH = "godot_index"

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    INDEX_PATH,embeddings,allow_dangerous_deserialization = True
    )

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0
)

   
question = "How to use animation tool in godot engine"

# --- similarity search with scores ---
docs_and_scores = vectorstore.similarity_search_with_score(question, k=3)
   

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
print(answer.content,'\n',context_text)



