import numpy as np
from pymilvus import MilvusClient, DataType, Collection , connections

MAX_USER_ID_LENGTH = 512
MAX_CHAT_ID_LENGTH = 512
MAX_MESSAGE_LENGTH = 2048
MAX_SESSION_JSON_LENGTH = 8192

MAX_ROLE_LENGTH = 16
MAX_CONTENT_LENGTH = 4096
MAX_WINDOW_JSON_LENGTH = 16384
MAX_PK_LENGTH = 1024

class Milvus:
    def __init__(self, config):
        self.config = config
        self.client = MilvusClient(uri=config['uri'])
        self.database = self.create_database(config["db_name"])

        self.create_session_memory_collection()
        self.create_chat_logs_collection()
        self.create_context_window_collection()

    

    def create_database(self, database_name):
        if database_name not in self.client.list_databases():
            self.client.create_database(database_name)
            print(f"Database '{database_name}' created.")
        else:
            print(f"Database '{database_name}' already exists.")
    
    def ensure_loaded(self, collection_name: str):
        # Milvus cần load collection trước khi query/get/search
        try:
            self.client.load_collection(collection_name=collection_name)
        except Exception:
            # nếu version milvus_client không có load_collection, fallback ORM
            connections.connect(alias="default", host="localhost", port="19530")
            Collection(collection_name).load()


    def retrieve_relevant_session_memory(self, query_embedding: list, top_k: int =5):
        search_params = {
            "metric_type": self.config['metric_type'],
            "params": {"nprobe": self.config['nprobe']}
        }
        results = self.client.search(
            collection_name=self.config['session_collection_name'],
            data=query_embedding,
            limit=top_k,
            search_params=search_params,
            output_fields=["user_id", "chat_id", "session_content", "full_session_json"]
        )
        return results

    def make_pk(self, user_id: str, chat_id: str, session_id: int) -> str:
        return f"{user_id}::{chat_id}::{session_id}"

    def create_session_memory_collection(self):
        if self.client.has_collection(self.config["session_collection_name"]):
            print(f"Collection '{self.config['session_collection_name']}' already exists.")
            return
        # 1) Schema

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("pk", DataType.VARCHAR, max_length=MAX_USER_ID_LENGTH + MAX_CHAT_ID_LENGTH + 20, is_primary=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=MAX_USER_ID_LENGTH)
        schema.add_field("chat_id", DataType.VARCHAR, max_length=MAX_CHAT_ID_LENGTH)
        schema.add_field("session_id", DataType.INT64)
        schema.add_field("session_content", DataType.VARCHAR, max_length=MAX_MESSAGE_LENGTH)
        schema.add_field("full_session_json", DataType.VARCHAR, max_length=MAX_SESSION_JSON_LENGTH)
        
        # vector field
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.config['embedding_dimension'])
        # 2) index 
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=self.config['index_type'],
            metric_type=self.config['metric_type'],
            params={"nlist": self.config['nlist']}
        )


        self.client.create_collection(
            collection_name=self.config["session_collection_name"],
            schema=schema,
            index_params=index_params,
        )
        print(f"Collection '{self.config['session_collection_name']}' created.")
    
    # create collection to save current message window 

    def create_context_window_collection(self):
        name = self.config["context_window_collection_name"]
        if self.client.has_collection(name):
            print(f"Collection '{name}' already exists.")
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("pk", DataType.VARCHAR, max_length=MAX_PK_LENGTH, is_primary=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=MAX_USER_ID_LENGTH)
        schema.add_field("chat_id", DataType.VARCHAR, max_length=MAX_CHAT_ID_LENGTH)

        schema.add_field("current_context_length", DataType.INT64)
        schema.add_field("lastest_summary_idx", DataType.INT64)
        schema.add_field("window_json", DataType.VARCHAR, max_length=MAX_WINDOW_JSON_LENGTH)
        schema.add_field("latest_summary_json", DataType.VARCHAR, max_length=MAX_SESSION_JSON_LENGTH)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="pk",
            index_type="INVERTED",   # scalar index for VARCHAR
        )

        self.client.create_collection(
            collection_name=self.config["context_window_collection_name"],
            schema=schema,
            index_params=index_params,
)

        print(f"Collection '{name}' created.")

    # create collection to save all chat logs
    def create_chat_logs_collection(self):
        name = self.config["chat_logs_collection_name"]
        if self.client.has_collection(name):
            print(f"Collection '{name}' already exists.")
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("pk", DataType.VARCHAR, max_length=MAX_PK_LENGTH, is_primary=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=MAX_USER_ID_LENGTH)
        schema.add_field("chat_id", DataType.VARCHAR, max_length=MAX_CHAT_ID_LENGTH)
        schema.add_field("idx", DataType.INT64)
        schema.add_field("role", DataType.VARCHAR, max_length=MAX_ROLE_LENGTH)  # "user" | "assistant"
        schema.add_field("content", DataType.VARCHAR, max_length=MAX_CONTENT_LENGTH)
        schema.add_field("created_at", DataType.INT64)  # epoch seconds

        # Optional: embed each message for semantic search (nếu muốn)
        # schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.config["embedding_dimension"])
        # index_params = MilvusClient.prepare_index_params()
        # index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="pk", index_type="INVERTED")
        index_params.add_index(field_name="idx", index_type="STL_SORT")  # optional, for range/sort

        self.client.create_collection(
            collection_name=self.config["chat_logs_collection_name"],
            schema=schema,
            index_params=index_params,
        )

        print(f"Collection '{name}' created.")


    def insert(self, collection_name, data):
        """
            Chèn dữ liệu vào collection
        """
        return self.client.insert(
            collection_name=collection_name,
            data=data
        )
    def delete(self, collection_name, ids: list):
        """
        Delete items by their IDs.
        """
        collection_name = self.config["schema_config"]["collection_name"]
        try:
            return self.client.delete(collection_name=collection_name, ids=ids)
        except Exception:
            print("Delete failed.")
            raise

    def num_entities(self) -> int:
        coll = self.config["collection_name"]
        try:
            if self.client.has_collection(coll):
                connections.connect(alias="default", host="localhost", port="19530")
                collection = Collection(name=coll)
                return collection.num_entities
            else:
                print("Collection '%s' does not exist." % coll)
                return 0
        except Exception as e:
            print("Failed to get number of entities in collection '%s': %s" % (coll, e))
            return 0
        
        
        
