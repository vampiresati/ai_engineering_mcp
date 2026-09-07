import os
from pathlib import Path
from dotenv import load_dotenv
BASE_PATH = Path(__file__).resolve().parent

def get_pipeconeapi_env():
    pinecone_env_path = BASE_PATH / 'all_env_configrations/.pinecone_env'
    load_dotenv(dotenv_path=pinecone_env_path)
    pinecone_secret_key = os.getenv('PINECONE_API_KEY')
    return pinecone_secret_key



if __name__=='__main__':
    pinecone_secret_key=get_pipeconeapi_env()


