import os, sys, warnings, yaml
from typing import List, Any, Union, Optional

class Config:

    def __init__(self, app: 'App') -> None:

        self.app = app
        self.config_file = self.app.root + "/config.yml"
        self.data_dir = None

    def load_config_file(self) -> None:

        def process_dirs(dirs):

            for k, v in dirs.items():

                _key = k + '_dir'
                _value = self.app.root + '/' + v

                setattr(self, _key, _value)

                if k == 'data':
                    self.app.data_dir = _value

        if not os.path.exists(self.config_file):
            self.create_config_file(self.app.app_type)

        with open(self.config_file, 'r') as file:

            for k, v in yaml.safe_load(file).items():

                if k == 'dirs':
                    process_dirs(v)

                elif k == 'cloud':
                    self.app.cloud.process_config(v)

                else:
                    setattr(self.config, k, v)

    def create_config_file(self, file_base: Optional[str]=None) -> None:

        if not file_base:
            file_base = {"dirs": {"data": "data"}}

        if file_base == 'google':
            file_base = {"dirs": {"data": "data"}, "cloud": {"google": {"bucket_name": "sequenzia-public", "blob_dir": "projects/data", "cloud_files": "creditcard.csv.zip"}}}

        with open(self.config_file, 'w') as file:
            yml_doc = yaml.dump(file_base, file, default_flow_style=False, sort_keys=False)


