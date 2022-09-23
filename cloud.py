import os, sys, yaml, zipfile, pandas as pd
from typing import List, Dict, Any, Union, Optional
from google.cloud import storage
from google.cloud.storage import Bucket

class Cloud:

    def __init__(self, app: Any) -> None:

        self.app = app

    def process_config(self, config: Dict) -> None:

        for k, v in config.items():
            if k == 'google':
                self.google = self.Google(self.app, **v)

    class Google:

        def __init__(self, app: Any, **kwargs) -> None:

            self.app = app

            for k, v in kwargs.items():
                setattr(self, k, v)

            self.storage_client_anonymous = storage.Client.create_anonymous_client()
            self.local_files = []
            self.datasets = []

            self.setup_files()


        def setup_files(self) -> None:

            bucket = self.storage_client_anonymous.bucket(self.bucket_name)

            for fn in self.cloud_files:
                blob_name = self.blob_dir+'/'+fn
                self.local_files.append(self.download_save_file(fn, self.app.data_dir, bucket, blob_name))

            return

        def download_save_file(self, zip_fn: str, data_dir: str, bucket: Bucket, blob_name: str) -> str:

            fn = zip_fn.rstrip('.zip')
            fp = data_dir + '/' + fn
            zip_fp = data_dir + '/' + zip_fn

            blob = bucket.blob(blob_name)

            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            if not os.path.exists(fp):
                if not os.path.exists(zip_fp):
                    blob.download_to_filename(zip_fp)

                with zipfile.ZipFile(zip_fp, 'r') as zip_file:
                    zip_file.extractall(data_dir)

            return fp