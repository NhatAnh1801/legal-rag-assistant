from .base import BaseDataLoader
from huggingface_hub import snapshot_download

import pandas as pd
import duckdb
import os

VN_BASE_PATH = "./data/raw/vn/vietnamese_legal_documents"

class VietnameseDataLoader(BaseDataLoader):
    def __init__(self):
        if not os.path.exists(VN_BASE_PATH):
            self.download()
    
    def download(self):
        snapshot_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            repo_type="dataset",
            local_dir=VN_BASE_PATH
        )
        
    def load(self, batch_size: int=None, offset: int=None) -> pd.DataFrame:
        con = duckdb.connect()
        query = f"""
            SELECT 
                m.id,
                m.title,
                m.so_ky_hieu,
                m.loai_van_ban,
                m.ngay_ban_hanh,
                m.co_quan_ban_hanh,
                m.linh_vuc,
                m.nganh,
                m.ngay_co_hieu_luc,
                m.ngay_het_hieu_luc,
                m.nguoi_ky,
                m.pham_vi,
                c.content_html
            FROM read_parquet('{VN_BASE_PATH}/data/metadata.parquet') m
            JOIN(
                SELECT DISTINCT id, content_html
                FROM read_parquet('{VN_BASE_PATH}/data/content.parquet')
                WHERE content_html IS NOT NULL
            ) c
            ON CAST(m.id AS VARCHAR) = c.id
            WHERE c.content_html IS NOT NULL AND m.tinh_trang_hieu_luc = 'Còn hiệu lực'
            ORDER BY m.id
        """
        if batch_size is not None:
            query += f" LIMIT {batch_size}"
        if offset is not None:
            query += f" OFFSET {offset}"
        return con.execute(query).df()

    def total_rows(self) -> int:
        con = duckdb.connect()
        query = f"""
            SELECT COUNT(*) FROM read_parquet('{VN_BASE_PATH}/data/metadata.parquet')
        """
        return con.execute(query).fetchone()[0]
        