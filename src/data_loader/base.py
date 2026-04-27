from abc import ABC, abstractmethod
import pandas as pd

class BaseDataLoader(ABC):
    @abstractmethod
    def download(self):
        """Download dataset"""
        pass

    @abstractmethod
    def load(self, batch_size: int, offset: int) -> pd.DataFrame:
        """Process each batch"""
        pass