from src.data_loader.base import BaseDataLoader
from src.data_loader.vn import VietnameseDataLoader
from src.processing.base import BaseDocumentProcessor
from src.processing.vn import VietnameseDocumentProcessor
from src.qdrant import Qdrant
from src.models.embeddings.gte_multi_base import GTE 

from dataclasses import dataclass
from typing import Any, Type

@dataclass(frozen=True)
class CountryConfig:
    loader_class: Type[BaseDataLoader]
    processor_class: Type[BaseDocumentProcessor]
    embedding_model: Any
    collection_name: str
    hash_cache_path: str
    vector_db_class: Type[Any]
    id_field: str
    content_field: str
    metadata_fields: dict[str, Any]
    
COUNTRY_CONFIG = {
    "vn": CountryConfig(
        loader_class=VietnameseDataLoader,
        processor_class=VietnameseDocumentProcessor,
        embedding_model=GTE(),
        collection_name="vn_documents",
        hash_cache_path="./data/cache/vn_hashes.json",
        vector_db_class=Qdrant,
        id_field="id",
        content_field="content_html",
        metadata_fields={
            "title": None,
            "so_ky_hieu": None,
            "loai_van_ban": None,
            "ngay_ban_hanh": None,
            "co_quan_ban_hanh": None,
            "linh_vuc": None,
            "nganh": None,
            "ngay_co_hieu_luc": None,
            "ngay_het_hieu_luc": None,
            "nguoi_ky": None,
            "pham_vi": None,
        }
    )
}

def get_country_config(country: str) -> CountryConfig:
    return COUNTRY_CONFIG[country]