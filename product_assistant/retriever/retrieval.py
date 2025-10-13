import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from product_assistant.utils.model_loader import ModelLoader
from product_assistant.utils.config_loader import load_config
from dotenv import load_dotenv

class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vs = None
        self.retriever = None

    def _load_env_variables(self):
        '''
        Load environment variables from .env file.
        '''
        load_dotenv()
        required_vars = [
            'GROQ_API_KEY', 'OPENAI_API_KEY', 
            'ASTRA_DB_API_ENDPOINT', 'ASTRA_DB_APPLICATION_TOKEN', 'ASTRA_DB_KEYSPACE'
        ]

        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')
        
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.astra_db_api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
        self.astra_db_application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
        self.astra_db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    def load_retriever(self):
        if not self.vs:
            collection_name = self.config['astra_db']['collection_name']

            self.vs = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.astra_db_api_endpoint,
                token=self.astra_db_application_token,
                namespace=self.astra_db_keyspace,
            )

        if not self.retriever:
            top_k = self.config['retriever']['top_k'] if 'retriever' in self.config else 3
            retriever = self.vs.as_retriever(search_kwargs={"k": top_k})
            print("Retriever loaded successfully.")
            return retriever

    def call_retriever(self, query):
        retriever = self.load_retriever()
        output = retriever.invoke(query) #type: ignore
        return output
