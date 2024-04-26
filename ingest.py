import time

print("testing test.py")

time.sleep(5)


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS







DATA_PATH = "uploaded_files"
DB_FAISS_PATH = "databases/new_database/"


def create_vectordb_and_merge():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    texts = text_splitter.split_documents(documents)



    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5',
                                       model_kwargs={'device': 'cuda'}))

    db = FAISS.from_documents(texts, embeddings)

    db1 = FAISS.load_local('databases/main_database/700-15', embeddings)
    db1.merge_from(db)
    
    db.save_local(DB_FAISS_PATH)

    


if __name__ == '__main__':
    create_vectordb_and_merge()