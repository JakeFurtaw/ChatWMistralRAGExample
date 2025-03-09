from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.retrievers import BaseRetriever

SYSTEM_PROMPT = """
You are a helpful AI Assistant that is amazing at coding. You have extensive knowledge about various old and new 
coding languages. When you generate your response make sure you talk like a pirate.
"""


llm = Ollama(model = "llama3.3",
             temperature=.7)

chat_engine = SimpleChatEngine.from_defaults(
    #retriever=BaseRetriever,
    llm=llm,
    system_prompt=SYSTEM_PROMPT,
)

def main():
    while (query:=input("Enter your question: ")) != "e":
        llm_response=chat_engine.stream_chat(query)
        for token in llm_response:
            print(token, end="")


if __name__ == "__main__":
    main()