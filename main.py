import json
from typing import List

# unstructured
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# langchain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

import os

# load .env
from dotenv import load_dotenv

# all initializations ----

# env
load_dotenv()


# api keys
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

# embedding
embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
    )

# pinecone
index_name="universitydb"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)



def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning document : {file_path}")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )

    print(f"Extract {len(elements)} elements")
    return elements

# chunking by titile 
def create_chucks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print(" creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )

    print(f"created {len(chunks)} chunks")
    return chunks

def text_extract(chunks):
    docs=[]
    for chunk in chunks :
        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source":chunk.metadata.filename,
                    "original_docs":chunk.metadata.orig_elements,
                    "pages":chunk.metadata.page_number
                }
            )
        )
    
    return docs

def vectorise(document):

    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    if index.describe_index_stats()["total_vector_count"]==0:
    vector_store.add_documents(document)


    vector_store.from_documents
    print("vectors created sucessfully")





    





def main():
    file_path="./docs/EJ1172284.pdf"
    elements= partition_document(file_path)

    elements[0].to

    # chunking by titile
    chunks=create_chucks_by_title(elements)

    # extract text and meta data 
    langchain_document=text_extract(chunks)

    # vector db
    if pc.has_index(index_name):
        vectorise(langchain_document)
    
    vector_store.as_retriever

    













if __name__ == "__main__":
    main()