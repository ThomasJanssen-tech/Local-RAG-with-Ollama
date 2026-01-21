#################################################################################################################################################################
###############################   1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ############################################################################
#################################################################################################################################################################

from dotenv import load_dotenv
import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


pd.options.mode.chained_assignment = None

load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("describe React's rendering lifecycle"), k=5
)

for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")