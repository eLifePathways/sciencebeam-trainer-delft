import logging
import os
from typing import Optional

from delft.textClassification.models import getModel

from delft.textClassification.wrapper import (
    Classifier as _Classifier
)

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_trainer_delft.embedding.manager import EmbeddingManager

from sciencebeam_trainer_delft.text_classification.config import ModelConfig
from sciencebeam_trainer_delft.text_classification.saving import (
    ModelSaver,
    ModelLoader
)


LOGGER = logging.getLogger(__name__)


DEFAULT_EMBEDDINGS_PATH = 'delft/resources-registry.json'


class Classifier(_Classifier):
    def __init__(
        self,
        download_manager: Optional[DownloadManager] = None,
        embedding_registry_path: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        **kwargs
    ):
        self.embedding_registry_path = embedding_registry_path or DEFAULT_EMBEDDINGS_PATH
        if download_manager is None:
            download_manager = DownloadManager()
        if embedding_manager is None:
            embedding_manager = EmbeddingManager(
                path=self.embedding_registry_path,
                download_manager=download_manager
            )
        self.download_manager = download_manager
        self.embedding_manager = embedding_manager
        super().__init__(**kwargs)

    def save_to(self, model_path: str):
        # create subfolder for the model if not already exists

        saver = ModelSaver(model_config=self.model_config)
        saver.save_model_config(
            self.model_config,
            os.path.join(model_path, self.config_file)
        )

        if self.model_config.fold_number == 1:
            if self.model is not None:
                saver.save_model_weights(
                    self.model,
                    os.path.join(
                        model_path,
                        self.model_config.architecture + "." + self.weight_file
                    )
                )
            else:
                LOGGER.error('Model has not been built')
        else:
            if self.models is None:
                LOGGER.error('nfolds models have not been built')
            else:
                for i in range(0, self.model_config.fold_number):
                    saver.save_model_weights(
                        self.models[i],
                        os.path.join(
                            model_path,
                            self.model_config.architecture + ".model{0}_weights.hdf5".format(i)
                        )
                    )
                LOGGER.info('nfolds model saved')

    def get_embedding_for_model_config(self, model_config: ModelConfig):
        embedding_name = model_config.embeddings_name
        embedding_name = self.embedding_manager.ensure_available(embedding_name)
        LOGGER.info('embedding_name: %s', embedding_name)
        embeddings = self.embedding_manager.get_embeddings_for_name(
            embedding_name
        )
        if embeddings.embed_size <= 0:
            raise AssertionError(
                'invalid embedding size, embeddings not loaded? %s' % embedding_name
            )
        return embeddings

    def load_from(self, model_path: str):
        loader = ModelLoader(download_manager=self.download_manager)
        self.model_config = loader.load_model_config_from_file(
            os.path.join(model_path, self.config_file)
        )

        # load embeddings
        self.embeddings = self.get_embedding_for_model_config(self.model_config)
        self.model_config.word_embedding_size = self.embeddings.embed_size

        self.model = getModel(self.model_config, self.training_config)
        if self.model_config.fold_number == 1:
            loader.load_model_weights_from_file(
                os.path.join(
                    model_path,
                    self.model_config.architecture + "." + self.weight_file
                ),
                self.model
            )
        else:
            self.models = []
            for i in range(0, self.model_config.fold_number):
                local_model = getModel(self.model_config, self.training_config)
                loader.load_model_weights_from_file(
                    os.path.join(
                        model_path,
                        self.model_config.architecture + ".model{0}_weights.hdf5".format(i)
                    ),
                    local_model
                )
                self.models.append(local_model)
