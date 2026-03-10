import json
import sys
import random
from typing import Iterable, Iterator, Any , Union
from pathlib import Path

from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success


class FilesDataLoader:
    def __init__(self, Files: list[Path], settings: dict):
        """
        Files: データファイルのパス
        settings: 設定辞書
        """
        self.files = Files
        self.batch_size = settings.get("batch_size", 1)
        self.shuffle = settings.get("shuffle", False)
        self.start_idx = settings.get("start_idx", 0)

    def __iter__(self) -> Iterator[list[Any]]:
        """
        バッチごとにデータをイテレートするジェネレーター
        Yields:
            batch: バッチデータのリスト (jsonの場合は辞書のリスト、テキストファイルの場合はファイルのリスト)
        """
        if self.shuffle:
            random.shuffle(self.files)
        for i in range(self.start_idx, len(self.files), self.batch_size):
            print(msg_success(f"DataLoader: Processing items {i} to {i + self.batch_size}..."))
            batch = self.files[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch, i, min(i + self.batch_size, len(self.files))

    def __len__(self) -> int:
        n = len(self.files) // self.batch_size
        if not self.drop_last and len(self.files) % self.batch_size != 0:
            n += 1
        return n

    def _load_data(self, datapath: Union[str, Path]) -> Path:
        datapath = Path(datapath)
        print(f'datapath: {datapath} {datapath.is_file()} {datapath.is_dir()}')
        if datapath.is_file():
            print(msg_info(f"DataLoader: Loading file {datapath}"))
            return self._load_data_file(datapath)

        elif datapath.is_dir():
            print(msg_info(f"DataLoader: Loading all files in directory {datapath}"))
            all_data = []
            for file_path in datapath.glob("*"):
                file_data = self._load_data_file(file_path)
                if file_data is not None:
                    all_data.append(file_data)
            return all_data
        else:
            raise ValueError("DataLoader: datapath must be a string or Path object.")
            return None

    def _load_data_file(self, datapath: Path) -> list[Any]:
        if isinstance(datapath, str):
            datapath = Path(datapath)

        # # データパスがフォルダの場合はglobでファイルを取得
        # if datapath.is_dir():
        #     print_error(f"DataLoader: data must be a file, not a directory: \n{datapath}")
        #     sys.exit(1)

        if datapath.suffix == ".json":
            print(msg_info(f"DataLoader: Loading JSON file {datapath}"))
            with open(datapath, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif datapath.suffix == ".jsonl":
            print(msg_info(f"DataLoader: Loading JSONL file {datapath}"))
            data = []
            with open(datapath, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif datapath.suffix in [".md",".txt", ".tsv", ".csv"]:  # 未対応
            # print_info(f"DataLoader: Loading text file {datapath}")
            # ファイル全体を読み取る
            with open(datapath, "r", encoding="utf-8") as f:
                data = f.read()

            # csv、tsvの場合、pandasか何かで処理が必要
        else:
            data = None
            print(msg_error(f"DataLoader: Unsupported file format `{datapath.suffix}`. Supported formats are .md .txt .json, .jsonl, Not yet supported: .tsv, .csv"))
        return data
    


class ListDataLoader:
    def __init__(self, data: list[dict], settings: dict):
        """
        Files: データファイルのパス
        settings: 設定辞書
        """
        self.data = data
        self.batch_size = settings.get("batch_size", 1)
        self.shuffle = settings.get("shuffle", False)
        self.start_idx = settings.get("start_idx", 0)
        self.drop_last = settings.get("drop_last", False)

    def __iter__(self) -> Iterator[list[Any]]:
        """
        バッチごとにデータをイテレートするジェネレーター
        Yields:
            batch: バッチデータのリスト (jsonの場合は辞書のリスト、テキストファイルの場合はファイルのリスト)
        """
        if self.shuffle:
            random.shuffle(self.data)
        for i in range(self.start_idx, len(self.data), self.batch_size):
            print(msg_success(f"DataLoader: Processing items {i} to {i + self.batch_size}..."))
            batch = self.data[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch, i, min(i + self.batch_size, len(self.data))

    def __len__(self) -> int:
        n = len(self.data) // self.batch_size
        if not self.drop_last and len(self.data) % self.batch_size != 0:
            n += 1
        return n
