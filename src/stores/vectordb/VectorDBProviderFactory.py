from .providers import QdrantDBProvider,PGVectorProvider
from controllers.BaseController import BaseController
from .VectorDBEnum import VectorDBEnum
from sqlalchemy.orm import sessionmaker
class VectorDBProviderFactory:
    def __init__(self,config):
        self.config = config
        self.base_controller = BaseController()
    def create(self,provider:str,db_client:sessionmaker=None):
        if provider == VectorDBEnum.QDRANT.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)
            return QdrantDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD
            )
        elif provider == VectorDBEnum.PGVECTOR.value:
            return PGVectorProvider(
                db_client=db_client,
                default_vector_size=self.config.EMBEDDING_MODEL_SIZE,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                index_threshold=self.config.VECTOR_DB_PGVEC_INDEX_THRESHOLD
            )
        return None