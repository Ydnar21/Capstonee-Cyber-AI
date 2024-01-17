from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema.document import Document
from typing import List


directory = ''
os.chdir(directory)

result = load_dotenv('keys.env', verbose=True)

if result:
    print("environment file specified")
else:
    print("you do not have an env file with the api key. this wont work")

OPENAI_API_KEY = os.getenv('key')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# workaraound to have relevance scores. Inspired by top solution from: https://github.com/langchain-ai/langchain/issues/5067
class MyVectorStoreRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
        )

        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs

meta_list = ['', 'trojans', 'rootkits', 'ransomware', 'spyware', 'worms', 'bots', 'evasion', 'infection', 'persistance', 'process injection', 'system components', 'ICS SCADA']

vectordb = Chroma(persist_directory="database", embedding_function=embeddings)
model_name = 'gpt-3.5-turbo'
model = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=OPENAI_API_KEY)

prompt_template = """You are a malware expert that answers in very high detail. 
{summaries}
Question: {question}
Answer: 
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "summaries"])
chain_type_kywargs2 = {"prompt": PROMPT}

while True:
    test_type = input(str("would you like to do the general test or the filtered test? answer general or filter\n"))
    if test_type == "general" or test_type == "filter":
        break
    else:
        print()
        print("please answer 'general' or 'filter'. no capital letters")
        print()
if test_type == "general":
    tester = RetrievalQAWithSourcesChain.from_chain_type(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'k':10}), return_source_documents = True, chain_type_kwargs=chain_type_kywargs2, max_tokens_limit=4090)
    print("starting general search test.")
    print()
    print("type exit to finish the program")
    print()
    while True:
        query = input(str("Human:"))
        if query == "exit":
            break
        else:
            try:
                result = tester({"question": query}, return_only_outputs=True)
                print()
                documents = result["source_documents"]
                position = 1
                for res in documents:
                    print(f"document # {position}:")
                    print(f"distance is: {res.metadata['score']}")
                    print(f"source = {res.metadata['source']}")
                    print(f"topic = {res.metadata['topic']}")
                    print(res.page_content)
                    print()
                    print()
                    print()
                    position += 1
                score = 0
                for res in documents:
                    value = res.metadata['score']
                    score += value
                final_score = score/(len(documents))
                print(f"average score is: {final_score}")
                print()
                print("answer:")
                print(result["answer"])
                print()
            except Exception as e:
                print(f'error is {e}')
                print()

            
                
else:
    print("starting filter search test.")
    while True:
        print(meta_list)
        topic_filter = input(str("type a topic from the list above\n"))
        if topic_filter in meta_list:
            break
        else:
            print()
            print("please type a topic from the list.")
            print()
    tester = RetrievalQAWithSourcesChain.from_chain_type(llm=model, retriever=MyVectorStoreRetriever(vectorstore=vectordb, search_type="similarity", search_kwargs={'filter': {'topic': str(topic_filter)},'k': 10}), return_source_documents = True, chain_type_kwargs=chain_type_kywargs2, max_tokens_limit=4090)
    print("type exit to finish the program")
    print()
    while True:
        query = input(str("Human:"))
        if query == "exit":
            break
        else:
            try:
                result = tester({"question": query}, return_only_outputs=True)
                print()
                documents = result["source_documents"]
                position = 1
                for res in documents:
                    print(f"document # {position}:")
                    print(f"distance is: {res.metadata['score']}")
                    print(f"source = {res.metadata['source']}")
                    print(f"topic = {res.metadata['topic']}")
                    print(res.page_content)
                    print()
                    print()
                    print()
                    position += 1
                score = 0
                for res in documents:
                    value = res.metadata['score']
                    score += value
                final_score = score/(len(documents))
                print(f"average score is: {final_score}")
                print()
                print("answer:")
                print(result["answer"])
                print()
            except Exception as e:
                print(f'error is {e}')
                print()
            
