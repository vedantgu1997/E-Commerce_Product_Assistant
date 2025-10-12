import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from product_assistant.utils.model_loader import ModelLoader
from product_assistant.utils.config_loader import load_config

class DataIngestion:
    '''
    Class to handle data transformations and ingestion into AstraDB vector store.
    '''
    def __init__(self):
        '''
        Initialize the DataIngestion class.
        '''
        print("Initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = load_config()

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

    def _get_csv_path(self):
        '''
        Get the path to the CSV file containing product reviews.
        '''
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, 'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'CSV file not found at path: {csv_path}')

        return csv_path

    def _load_csv(self):
        '''
        Load the CSV file into a pandas DataFrame.
        '''
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id', 'product_title', 'rating', 'total_reviews', 'price', 'top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f'CSV file is missing required columns. Expected columns: {expected_columns}')

        return df

    def transform_data(self):
        '''
        Transform the data for ingestion.
        '''
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                'product_id': row['product_id'],
                'product_title': row['product_title'],
                'rating': row['rating'],
                'total_reviews': row['total_reviews'],
                'price': row['price'],
                'top_reviews': row['top_reviews']
            }
            product_list.append(product_entry)

        documents = []

        for entry in product_list:
            metadata = (
                f"product_id: {entry['product_id']}\n"
                f"title: {entry['product_title']}\n"
                f"rating: {entry['rating']}\n"
                f"total_reviews: {entry['total_reviews']}\n"
                f"price: {entry['price']}\n"
                f"top_reviews: {entry['top_reviews']}\n"
            )
            doc = Document(page_content=entry['top_reviews'], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents for ingestion.")
        return documents


    def store_in_vector_db(self, documents: List[Document]):
        '''
        Store the transformed data into AstraDB vector store.
        '''
        collection_name = self.config['astra_db']['collection_name']
        vs = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.astra_db_api_endpoint,
            token=self.astra_db_application_token,
            namespace=self.astra_db_keyspace
        )

        inserted_ids = vs.add_documents(documents)
        print(f"Inserted {len(inserted_ids)} documents into AstraDB collection '{collection_name}'.")
        return vs, inserted_ids

    def run_pipeline(self):
        '''
        Run the complete data ingestion pipeline.
        '''
        documents = self.transform_data()
        vs, _ = self.store_in_vector_db(documents)

        query = "Can you tell me the low budget iphone?"
        results = vs.similarity_search(query)

        print(f'Sample search results for query "{query}":')

        for res in results:
            print(f'Content: {res.page_content}\nMetadata: {res.metadata}\n')

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()

