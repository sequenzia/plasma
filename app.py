import os, sys, warnings, yaml
import pandas as pd
from typing import List, Any, Union, Optional
from dataclasses import dataclass
from pathlib import Path
from .cloud import Cloud


class App:

    def __init__(self, root: Optional[str] = None) -> None:

        if not root:
            root = os.getcwd()

        self.root = root
        self.datasets = []

        self.setup_app()

    def setup_app(self) -> None:

        self.config = self.Config(self.root)
        self.cloud = Cloud(self)

        if not os.path.exists(self.config.file):
            self.create_config_file()

        self.process_config_file()

        if self.cloud.config:
            self.setup_datasets()
        else:
            warnings.warn('Cannot setup datasets')

        return

    def create_config_file(self):

        base_file = {'dirs': {'data': 'data'}}
        config_file = self.root + '/config.yml'

        with open(config_file, 'w') as file:
            documents = yaml.dump(base_file, file)

    def setup_datasets(self) -> None:

        for fn in self.cloud.google.local_files:
            self.datasets.append(pd.read_csv(fn))
        return

    def process_config_file(self) -> None:

        def process_dirs(dirs):
            for k, v in dirs.items():

                _key = k + '_dir'
                _value = self.root + '/' + v

                setattr(self.config, _key, _value)

                if k == 'data':
                    self.data_dir = _value

        with open(self.config.file, 'r') as file:

            config_file = yaml.safe_load(file)

            for k, v in config_file.items():

                if k == 'dirs':
                    process_dirs(v)

                elif k == 'cloud':
                    self.cloud.process_config(v)

                else:
                    setattr(self.config, k, v)

    @dataclass
    class Config:
        root: str

        def __post_init__(self):
            self.file = self.root + "/config.yml"