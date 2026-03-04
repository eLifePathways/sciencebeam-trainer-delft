import logging
from datetime import datetime
from abc import ABC
import json
import os
from typing import Optional

from tf_keras.models import Model

from sciencebeam_trainer_delft.utils.cloud_support import auto_upload_from_local_file
from sciencebeam_trainer_delft.utils.io import open_file, write_text

from sciencebeam_trainer_delft.text_classification.config import ModelConfig
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager


LOGGER = logging.getLogger(__name__)


class _BaseModelSaverLoader(ABC):
    config_file = 'config.json'
    weight_file = 'model_weights.hdf5'
    meta_file = 'meta.json'

    def get_model_weights_filename(self, model_path: str, model_config: ModelConfig):
        return os.path.join(
            model_path,
            model_config.architecture + "." + self.weight_file
        )


class ModelSaver(_BaseModelSaverLoader):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def save_model_config(self, model_config: ModelConfig, filepath: str):
        LOGGER.debug('model_config: %s', model_config)
        with open_file(filepath, 'w') as fp:
            model_config.save_fp(fp)
        LOGGER.info('model config file saved to %s', filepath)

    def save_model_weights(self, model: Model, filepath: str):
        with auto_upload_from_local_file(filepath) as local_filepath:
            model.save(local_filepath)
        LOGGER.info('model saved to %s', filepath)

    def save_meta(self, meta: dict, filepath: str):
        write_text(
            filepath,
            json.dumps(meta, sort_keys=False, indent=4)
        )
        LOGGER.info('model meta saved to %s', filepath)

    def update_checkpoints_meta_file(self, filepath: str, checkpoint_directory: str, epoch: int):
        try:
            with open_file(filepath, 'r') as fp:
                meta = json.load(fp)
        except FileNotFoundError:
            meta = {}
        checkpoint_meta = {
            'epoch': (1 + epoch),
            'path': checkpoint_directory,
            'timestamp': datetime.utcnow().isoformat()
        }
        meta['checkpoints'] = meta.get('checkpoints', [])
        meta['checkpoints'].append(checkpoint_meta)
        meta['last_checkpoint'] = checkpoint_meta
        with open_file(filepath, 'w') as fp:
            json.dump(meta, fp, sort_keys=False, indent=4)
        LOGGER.info('updated checkpoints meta: %s', filepath)

    def save_to(self, directory: str, model: Model, meta: Optional[dict] = None):
        self.save_model_config(self.model_config, os.path.join(directory, self.config_file))
        self.save_model_weights(
            model,
            self.get_model_weights_filename(directory, model_config=self.model_config)
        )
        if meta:
            self.save_meta(meta, os.path.join(directory, self.meta_file))

    def add_checkpoint_meta(self, checkpoint_directory: str, epoch: int):
        self.update_checkpoints_meta_file(
            os.path.join(os.path.dirname(checkpoint_directory), 'checkpoints.json'),
            checkpoint_directory=checkpoint_directory,
            epoch=epoch
        )


class ModelLoader(_BaseModelSaverLoader):
    def __init__(
        self,
        download_manager: Optional[DownloadManager] = None
    ):
        if download_manager is None:
            download_manager = DownloadManager()
        self.download_manager = download_manager

    def load_model_config_from_file(self, filepath: str):
        LOGGER.info('loading model config from %s', filepath)
        with open_file(filepath, 'r') as fp:
            return ModelConfig.load_fp(fp)

    def load_model_weights_from_file(self, filepath: str, model: Model):
        LOGGER.info('loading model from %s', filepath)
        # we need a seekable file, ensure we download the file first
        local_filepath = self.download_manager.download_if_url(filepath)
        # using load_weights to avoid print statement in load method
        model.load_weights(local_filepath)
