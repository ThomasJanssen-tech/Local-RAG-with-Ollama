#################################################################################################################################################################
###############################   1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ############################################################################
#################################################################################################################################################################

from dotenv import load_dotenv
import os
import json
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import time


load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

###############################   DELETE CHROMA DB IF EXISTS AND INITIALIZE   ##################################################################################

if os.path.exists(os.getenv("DATABASE_LOCATION")):
    shutil.rmtree(os.getenv("DATABASE_LOCATION"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

###############################   INITIALIZE TEXT SPLITTER   ###################################################################################################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

#################################################################################################################################################################
###############################   2.  PROCESSING THE JSON RESPONSE LINE BY LINE   ###############################################################################
#################################################################################################################################################################

###############################   FUNCTION TO EXTRACT RESPONSE LINE BY LINE   ###################################################################################



def process_json_lines(file_path):
    """Process each JSON line and extract relevant information."""
    extracted = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            extracted.append(obj)

    return extracted

file_content = process_json_lines(os.getenv("DATASET_STORAGE_FOLDER")+"data.txt")


#################################################################################################################################################################
###############################   3.  CHUNKING, EMBEDDING AND INGESTION   #######################################################################################
##################################################################################################################################################################

for line in file_content:
    # Skip entries that have errors or don't have required fields
    if 'error' in line or 'url' not in line or 'raw_text' not in line or 'title' not in line:
        continue

    print(line['url'])

    texts = text_splitter.create_documents([line['raw_text']],metadatas=[{"source":line['url'], "title":line['title']}])

    uuids = [str(uuid4()) for _ in range(len(texts))]

    vector_store.add_documents(documents=texts, ids=uuids)