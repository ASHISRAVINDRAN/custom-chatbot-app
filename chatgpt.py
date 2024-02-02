import os
import time
from pprint import pprint
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI

class MITKOpenAI:
    OPENAI_API_KEY = ''
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    def __init__(self, path:str = "./data/",temperature = 0.6):
        print('in mITKOPENai')
        loader = DirectoryLoader(path, glob="*.pdf")
        self.index = VectorstoreIndexCreator().from_loaders([loader])
        self.llm = ChatOpenAI(openai_api_key= self.OPENAI_API_KEY, model="gpt-3.5-turbo-16k-0613", temperature=temperature)

    def get_response(self, query:str):
        t_start = time.time()
        print('Querying for: ', query)
        local_response = self.index.query(query)
        print(f'Local response [Total words: {len(local_response.split())}]:')
        pprint(local_response)
        # print('--' * 20)
        # Uncomment this section for global response with RAG.
        # combo_response = self.index.query(query, llm=self.llm)
        # print(f'Combined response [Total words: {len(combo_response.split())}]:')
        # pprint(combo_response)
        print(f'Elapsed: {time.time() - t_start} seconds')
        return local_response
