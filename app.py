import os, sys, warnings, yaml
import pandas as pd
from typing import List, Any, Union, Optional
from dataclasses import dataclass
from pathlib import Path
from .config import Config
from .cloud import Cloud

class App:

    def __init__(self, root: Optional[str] = None, app_type: Optional[str] = None) -> None:

        if not root:
            root = os.getcwd()

        if not app_type:
            app_type = "google"

        self.root = root
        self.app_type = app_type
        self.datasets = []

        self.setup_app()

    def setup_app(self) -> None:

        self.config = Config(self)
        self.cloud = Cloud(self)

        self.config.load_config_file()

        if self.cloud.config:
            self.setup_datasets()
        else:
            warnings.warn('Cannot setup datasets')

        return

    def setup_datasets(self) -> None:

        for fn in self.cloud.google.local_files:
            self.datasets.append(pd.read_csv(fn))
        return