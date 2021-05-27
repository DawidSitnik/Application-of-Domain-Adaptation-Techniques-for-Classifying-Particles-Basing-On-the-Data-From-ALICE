from typing import List
import pandas as pd


class DataLoader:

    def __init__(self, columns_to_drop: List[str]):
        self.columns_to_drop = columns_to_drop
        self.label_column = 'pdg_code'

    def _read_file_from_pickle(self, file_path: str) -> pd.core.frame.DataFrame:
        return pd.read_pickle(file_path).drop(columns=self.columns_to_drop)

    def _get_proper_pdg_codes(self, data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return data[~data['pdg_code'].isin([3312.0, 3112.0, 13.0, -13.0])]

    def _rename_negative_class_labels(self, data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        data[self.label_column] = data[self.label_column].apply(lambda x: -x if x < 0 else x)
        return data

    def _rename_class_labels(self, data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        def rename_pdgcode(x):
            if x == 11.0:
                return 0 #'elektron'
            if x == 211.0:
                return 1 #'pion'
            if x == 321.0:
                return 2 #'kaon'
            if x == 2212.0:
                return 3 #'proton'

        data[self.label_column] = data[self.label_column].apply(lambda x: rename_pdgcode(x))
        return data

    def load_file(self, file_name: str):
        assert type(file_name) == str, "file_name should be a string"

        file_path = file_name

        data = self._read_file_from_pickle(file_path)
        data = self._get_proper_pdg_codes(data)
        data = self._rename_negative_class_labels(data)
        data = self._rename_class_labels(data)

        return data