from langchain_community.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
import time


class MITKGPT4All:
    def __init__(self, file_path: str = "./data/data.txt")
        llm = GPT4All(
            model="./wizardlm-13b-v1.2.Q4_0.gguf",
            #model="./mistral-7b-openorca.Q4_0.gguf",
            max_tokens=8096
        )
        loader = UnstructuredPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        faiss_index = FAISS.from_documents(all_splits, GPT4AllEmbeddings())
        retriever = faiss_index.as_retriever()
        self.chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    def get_response(self, query: str):
        t_start = time.time()
        response = self.chain.run(query)
        print(f'Elapsed: {time.time() - t_start} seconds')
        return response
