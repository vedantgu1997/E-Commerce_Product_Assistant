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
        pass

    def _load_env_variables(self):
        '''
        Load environment variables from .env file.
        '''
        pass

    def _get_csv_path(self):
        '''
        Get the path to the CSV file containing product reviews.
        '''
        pass

    def _load_csv(self):
        '''
        Load the CSV file into a pandas DataFrame.
        '''
        pass

    def transform_data(self):
        '''
        Transform the data for ingestion.
        '''
        pass

    def store_in_vector_db(self):
        '''
        Store the transformed data into AstraDB vector store.
        '''
        pass

    def run_pipeline(self):
        '''
        Run the complete data ingestion pipeline.
        '''
        pass

