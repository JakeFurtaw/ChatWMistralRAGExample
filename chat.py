from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# from llama_index.core.chat_engine import SimpleChatEngine
# from llama_index.core.chat_engine.context import ContextChatEngine
# from llama_index.core.retrievers import BaseRetriever

DATA_PATH = "data"
SYSTEM_PROMPT = """
You are a helpful AI Assistant that is amazing at coding. You have extensive knowledge about various old and new 
coding languages. When you generate your response make sure you talk like a pirate.
"""

def load_data():
    doc = SimpleDirectoryReader(input_dir=DATA_PATH).load_data()
    return doc

llm = Ollama(model = "llama3.3",
             temperature=.7)

embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large-instruct")


index = VectorStoreIndex.from_documents(documents=load_data(),
                                        embed_model=embed_model)

# chat_engine = SimpleChatEngine.from_defaults(
#     #retriever=BaseRetriever,
#     llm=llm,
#     system_prompt=SYSTEM_PROMPT,
# )

chat_engine = index.as_chat_engine(
    llm=llm,
    chat_mode=ChatMode.CONTEXT,
    system_prompt = SYSTEM_PROMPT,
    context_prompt=("Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer \n"
                    "the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: ")
    )

def main():
    while (query:=input("Enter your question: ")) != "e":
        llm_response=chat_engine.stream_chat(query)
        for token in llm_response.response_gen:
            print(token, end="", flush=True)


if __name__ == "__main__":
    main()