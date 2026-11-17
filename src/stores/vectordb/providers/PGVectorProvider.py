from calendar import c
from curses import meta
import json
from tkinter import NO

from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnum import (PgVectorDistanceMethodEnums,
                            PgVectorIndexTypeEnums,PgVectorTableSchemeEnums,DistanceMethodEnums)
from models.db_schemes import RetrievedDocument
import logging
from sqlalchemy.sql import text as sql_text

class PGVectorProvider(VectorDBInterface):
    def __init__(self, db_client, default_vector_size: int = 384, distance_method: str = None,index_threshold:int=100):
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.pgvector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value
        self.logger = logging.getLogger('uvicorn')
        self.default_index_name = lambda collection_name: f'{collection_name}_vector_idx'
        self.index_threshold = index_threshold
        self.distance_method = None
        if distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = PgVectorDistanceMethodEnums.DOT.value
        else:
            self.distance_method = PgVectorDistanceMethodEnums.COSINE.value

    async def connect(self):
        async with self.db_client() as session:
            async with session.begin():
                await session.execute(sql_text(
                    'CREATE EXTENSION IF NOT EXISTS vector'
                ))
    async def disconnect(self):
        pass

    async def is_collection_exists(self, collection_name):
        record = None
        async with self.db_client() as session:
            async with session.begin():
                list_tables = sql_text('SELECT * FROM pg_tables WHERE tablename = :collection_name')
                results = await session.execute(list_tables,{'collection_name':collection_name})
                record = results.scalar_one_or_none()
        return record
    
    async def list_all_collection(self) -> list:
        records = []
        async with self.db_client() as session:
            async with session.begin():
                list_tables = sql_text(f'SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix')
                results = await session.execute(list_tables,{'prefix':f'{self.pgvector_table_prefix}%'})
                records = results.scalars().all()
        return records

    '''
    pg_tables is a built-in PostgreSQL view that lists tables available in the database.
    It contains columns such as: schemaname, tablename, tableowner, tablespace

    tablename is simply a column in pg_tables containing each table's name.
    '''
    async def get_collection_info(self, collection_name:str):
        async with self.db_client() as session:
            async with session.begin():
                table_info_query = sql_text("""
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename = :collection_name
                """)
                count_query = sql_text(
                    f'SELECT COUNT(*) FROM {collection_name}' 
                )

                table_info = await session.execute(table_info_query,{'collection_name':collection_name})
                record_count = await session.execute(count_query)

                table_info = table_info.fetchone()
                record_count = record_count.scalar_one()
                if not table_info:
                    return None

                return {
                    'table_info': dict(table_info._mapping),
                    'record_count': record_count
                }

    async def delete_collection(self, collection_name):
        async with self.db_client() as session:
            async with session.begin():
                delete_query = sql_text(
                    f'DROP TABLE IF EXISTS {collection_name}'
                )
                await session.execute(delete_query)
        return True

    async def create_collection(self, collection_name: str,
                                      embedding_size: int,
                                      do_reset: bool = False):
        
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)

        is_collection_existed = await self.is_collection_exists(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.info(f"Creating collection: {collection_name}")
            async with self.db_client() as session:
                async with session.begin():
                    create_sql = sql_text(
                        f'CREATE TABLE {collection_name} ('
                            f'{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY,'
                            f'{PgVectorTableSchemeEnums.TEXT.value} text, '
                            f'{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), '
                            f'{PgVectorTableSchemeEnums.METADATA.value} jsonb DEFAULT \'{{}}\', '
                            f'{PgVectorTableSchemeEnums.CHUNK_ID.value} integer, '
                            f'FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) REFERENCES chunks(chunk_id)'
                        ')'
                    )
                    await session.execute(create_sql)            
            return True

        return False

    async def insert_one(self, collection_name, text, vector, metadata = None, record_id = None):
        is_collection_existed = await self.is_collection_exists(collection_name = collection_name)
        if not is_collection_existed:
            self.logger.error(f'Can not insert new record to non-existed collection {collection_name}')
            return False
        if not record_id:
            self.logger.error('Can not insert new record without chunk_id')
            return False
        async with self.db_client() as session:
            async with session.begin():
                insert_query = sql_text(
                    f'INSERT INTO {collection_name}'
                    f'({PgVectorTableSchemeEnums.TEXT.value}, {PgVectorTableSchemeEnums.VECTOR.value}, '
                    f'{PgVectorTableSchemeEnums.METADATA.value}, {PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                    'VALUES (:text, :vector, :metadata, :chunk_id)'
                )
                await session.execute(insert_query,{
                    'text':text,
                    'vector':f'[ {",".join(map(str, vector))} ]',
                    'metadata':json.dumps(metadata,ensure_ascii=False) if metadata is not None else '{}',
                    'chunk_id':record_id
                })
        _ = await self.create_index(collection_name=collection_name)
        return True

    async def insert_many(self, collection_name, texts, vectors, metadatas = None, record_ids = None, batch_size = 50):
        is_collection_existed = await self.is_collection_exists(collection_name = collection_name)
        if not is_collection_existed:
            self.logger.error(f'Can not insert new record to non-existed collection {collection_name}')
            return False
        if not record_ids:
            self.logger.error('Can not insert new record without chunk_id')
            return False
        if not (len(vectors) == len(record_ids) == len(texts)):
            self.logger.error('invalid data items')
            return False
        if metadatas is None:
            metadatas = [None] * len(texts)
        elif len(metadatas) != len(texts):
            self.logger.error('Invalid metadata length')
            return False
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0,len(texts),batch_size):
                    text_batch = texts[i:i+batch_size]
                    vector_batch = vectors[i:i+batch_size]
                    metadata_batch = metadatas[i:i+batch_size]
                    record_batch = record_ids[i:i+batch_size]
                    values = []
                    for _text,_vector,_metadata,_record_id in zip(text_batch,vector_batch,metadata_batch,record_batch):
                        _metadata = json.dumps(_metadata,ensure_ascii=False) if _metadata is not None else '{}'
                        values.append({
                            'text':_text,
                            'vector': f'[ {",".join(map(str, _vector))} ]',
                            'metadata':_metadata,
                            'chunk_id': _record_id
                        })
                    batch_query = sql_text(
                        f'INSERT INTO {collection_name}'
                        f'({PgVectorTableSchemeEnums.TEXT.value}, {PgVectorTableSchemeEnums.VECTOR.value}, '
                        f'{PgVectorTableSchemeEnums.METADATA.value}, {PgVectorTableSchemeEnums.CHUNK_ID.value}) '
                        'VALUES (:text, :vector, :metadata, :chunk_id)'
                    )

                    await session.execute(batch_query,values)
        _ = await self.create_index(collection_name=collection_name)
        return True

    async def search_by_vector(self, collection_name, vector, limit):
        is_collection_existed = await self.is_collection_exists(collection_name = collection_name)
        if not is_collection_existed:
            self.logger.error(f'Can not search record in a non-existed collection {collection_name}')
            return False

        vector = f'[ {",".join(map(str, vector))} ]'
        async with self.db_client() as session:
            async with session.begin():
                search_query = sql_text(
                    f'SELECT {PgVectorTableSchemeEnums.TEXT.value} as text, '
                    f'1 - ({PgVectorTableSchemeEnums.VECTOR.value} <=> :vector) as score '
                    f'FROM {collection_name} '
                    'ORDER BY SCORE DESC '
                    f'LIMIT {limit}'
                )
                '''
                In pgvector:
                - <-> is Euclidean (L2) distance.
                - <=> is cosine distance.
                - <#> is negative inner product.
                '''
                result = await session.execute(search_query,{'vector':vector})
                records = result.fetchall()

                return [RetrievedDocument(text=record.text,score=record.score) for record in records]

    async def is_index_exists(self,collection_name:str)->bool:
        index_name = self.default_index_name(collection_name=collection_name)
        async with self.db_client() as session:
            async with session.begin():
                check_query = sql_text(
                    '''
                    SELECT 1 FROM pg_indexes
                    WHERE tablename = :collection_name
                    AND indexname = :index_name
                    '''
                )
                result = await session.execute(check_query,{'collection_name':collection_name,'index_name':index_name})
                return result.scalar_one_or_none() is not None

    async def create_index(self,collection_name:str,
                           index_type:str=PgVectorIndexTypeEnums.HNSW.value):
        is_index_exists = await self.is_index_exists(collection_name=collection_name)
        if is_index_exists:
            return False
        async with self.db_client() as session:
            async with session.begin():
                count_query = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                result = await session.execute(count_query)
                count = result.scalar()
                if count < self.index_threshold:
                    return False
                index_name = self.default_index_name(collection_name=collection_name)
                create_idx_query = sql_text(
                    f'''
                    CREATE INDEX {index_name} ON {collection_name}
                    USING {index_type} ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})
                    '''
                )
                await session.execute(create_idx_query)
                self.logger.info(f'END: Created vector index for collection: {collection_name}')
                return True

    async def reset_index(self,collection_name:str,
                        index_type:str=PgVectorIndexTypeEnums.HNSW.value):
        index_name = self.default_index_name(collection_name=collection_name)
        async with self.db_client() as session:
            async with session.begin(): 
                drop_query  = sql_text(
                    f'DROP INDEX IF EXISTS {index_name}'
                )
                await session.execute(drop_query)
        return self.create_index(collection_name = collection_name,index_type = index_type)
            
            
