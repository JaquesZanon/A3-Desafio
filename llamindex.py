from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import resolve_embed_model
from tqdm import tqdm
import pandas as pd
from utils import *
from llama_index.core.node_parser import SentenceSplitter
from tqdm.notebook import tqdm
tqdm.pandas()

dataset_df = pd.read_csv('df_total_sel.csv').sample(n=1500)
dataset_df=dataset_df.drop(columns=['text_embedding','time','image', 'previewLink','publishedDate',
                                    'ratingsCount','Price',	'Unnamed: 0','infoLink','summary','description',
                                    'authors', 'publisher'])

embed_model = resolve_embed_model("local:thenlper/gte-large") 
llm = OpenAI(api_key='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')

Settings.llm=llm
Settings.embed_model=embed_model

llama_documents = create_llamaindex_documents(dataset_df)

  
print(
    "\nThe LLM sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.LLM),
)

print(
    "\nThe Embedding model sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.EMBED),
)

# criando os nodes
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(llama_documents)
nodes = custom_embeddings(nodes)# pode demorar um poco


###### essa parte mongoDB ######
mongo_uri = "mongodb+srv://jaqueszanon:pgjuDkwIGzrTPRiW@cluster0.gf1tlse.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
if not mongo_uri:
  print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME="books"
COLLECTION_NAME="book_collection_2"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
#collection.delete_many({})

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name=DB_NAME, collection_name=COLLECTION_NAME, index_name="vector_index")
vector_store.add(nodes)
#################################

################################# essa para o chat
index_vector, index_summary = get_indices(carregar=True)

query = "which reviewer was more cruel in his criticism and what is his id?"
chat_bot(query, llm, index_vector)






