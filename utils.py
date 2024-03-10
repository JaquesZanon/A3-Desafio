import json
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

def create_llamaindex_documents(dataset_df):
    """This function creates a customized llmaindex type docuemnt

    Args:
        dataset_df (dataframe): pandas dataframe

    Returns:
        llamaindexdocuments: documents ins llmaindex format
    """

    documents_json = dataset_df.to_json(orient='records')
    documents_list = json.loads(documents_json)

    llama_documents = []

    for document in documents_list:

        document["Title"] = json.dumps(document["Title"])
        document["Id"] = json.dumps(document["Id"])
        document["score"] = json.dumps(document["score"])
        document["User_id"] = json.dumps(document["User_id"])
        document["text"] = json.dumps(document["text"])

        llama_document = Document(
            text=document["text"],
            metadata=document,
            excluded_llm_metadata_keys=["text", "publisher", "summary"],
            excluded_embed_metadata_keys=["text", "publisher", "description", "summary"],
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )

        llama_documents.append(llama_document)
    return llama_documents



from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
tqdm.pandas()

def custom_embeddings(nodes):
    """Get embeddings from gte-large model

    Args:
        nodes (llamaindex nodes): nodes created by the llamindex

    Returns:
        llamaindex nodes: same nodes, but now with the embedings
    """
    
    embedding_model = SentenceTransformer("thenlper/gte-large")
    def get_embedding(text: str) -> list[float]:
        if not text.strip():
            print("Attempted to get embedding for empty text.")
            return []
        embedding = embedding_model.encode(text)
        return embedding.tolist()


    for node in tqdm(nodes):
        
        # node_embedding = embed_model.get_text_embedding(
        #     node.get_content(metadata_mode="all")
        # )
        node_embedding = get_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding
    return nodes


import pymongo

def get_mongo_client(mongo_uri):
  """Establish connection to the MongoDB."""
  try:
    client = pymongo.MongoClient(mongo_uri)
    print("Connection to MongoDB successful")
    return client
  except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")
    return None


from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex

def data_mongodb(mongo_client, add=False, nodes=None):
    """Add nodes to mongo db atlas vector database

    Args:
        mongo_client (client): pre-configurated mongo client
        add (bool, optional): if you want to add. Defaults to False.
        nodes (llmaindex nodes, optional):nodes to add. Defaults to None.

    Returns:
        _type_: _description_
    """
    DB_NAME="books"
    COLLECTION_NAME="book_collection_2"
    vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name=DB_NAME, collection_name=COLLECTION_NAME, index_name="vector_index")
    if add == True:
        vector_store.add(nodes)
        
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)
    return index



from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core import StorageContext, load_index_from_storage

def get_indices(nodes=None, carregar=True):

    if carregar == True:
        storage_context = StorageContext.from_defaults(persist_dir="index_prs")
        index_vector = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(persist_dir="index_sum")
        index_summary = load_index_from_storage(storage_context)
    else:
        index_vector = VectorStoreIndex(nodes=nodes)
        index_vector.storage_context.persist(persist_dir="index_prs")

        index_summary = SummaryIndex(nodes=nodes)
        index_summary.storage_context.persist(persist_dir="index_sum")
    return index_vector, index_summary



from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer

def chat_bot(query, llm, index):
    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key='usuario',
    )


    chat_engine = index.as_chat_engine(
            chat_mode="openai",
            llm=llm,
            memory=chat_memory,
            tool_choice="query_engine_tool",
            context_prompt=(
                "you know everything about books and their reviews"
            ),
            verbose=True,
        )
    response = chat_engine.chat(query)
    return(print(response.response))