from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.nvidia import NVIDIA
from llama_parse import LlamaParse
import os, glob
from dotenv import load_dotenv
# from llama_index.core.chat_engine.context import ContextChatEngine

load_dotenv()

DATA_PATH = "data"
SYSTEM_PROMPT = """
You are a helpful AI Assistant that is an expert at looking through json files for data to help university students get around campus.
You are en expert at locating faculty and other staff information like office numbers, emails, and phone numbers. Locations of 
departments, university amenities, and other services. You are also an expert at finding information about different majors
students might be using and classes that are required for them. Please be clear and concise with your answers. 
"""

def load_and_parse_data():
    supported_extensions = [".pdf", ".docx", ".xlsx", ".csv", ".xml", ".html", ".json"]
    doc = []
    all_files = glob.glob(os.path.join(DATA_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    for file in all_files:
        # file_extension = os.path.splitext(file)[1].lower()
        # if "LLAMA_CLOUD_API_KEY" in os.environ and file_extension in supported_extensions:
        #     parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
        #     file_extractor = {file_extension: parser}
        #     doc.extend(
        #         SimpleDirectoryReader(input_files=[file], file_extractor=file_extractor).load_data())
        # else:
        doc.extend(SimpleDirectoryReader(input_files=[file]).load_data())
    return doc


#Ollama LLM's
ollama_llm = Ollama(model = "llama3.3:70b",
                    request_timeout=30.0,
                    temperature=.7,
                    context_window=110000, #Increase context window for models with larger context windows
                    json_mode=False,# Not sure what this does might turn responses to json format
                    # additional_kwargs={'num_output':500} #If you want to limit the output you can mess with this
)
#Nvidia NIM's
# nvidia_llm = NVIDIA(model=,
#                     # max_tokens=,
#                     temperature=.7,
#                     # top_p=, #Optional top_p control
#                     nvidia_api_key=os.getenv("NVIDIA_API_KEY") #Uncomment this after API Key is in the .env file
# )

embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large-instruct")

index = VectorStoreIndex.from_documents(documents=load_and_parse_data(),
                                        embed_model=embed_model)

chat_engine = index.as_chat_engine(
    llm=ollama_llm, # Switch this to nvidia_llm if you want to use a NIM
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
        print('\n'+('-'*60))
        print('\n')


if __name__ == "__main__":
    main()