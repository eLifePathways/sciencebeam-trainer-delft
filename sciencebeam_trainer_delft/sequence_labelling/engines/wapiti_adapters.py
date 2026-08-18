import logging
import tempfile
import os
from pathlib import Path
from typing import Iterable, IO, List, Optional, Set, Tuple

import numpy as np

from delft.sequenceLabelling.reader import (
    _translate_tags_grobid_to_IOB as translate_tags_grobid_to_IOB
)

from sciencebeam_trainer_delft.sequence_labelling.evaluation import ClassificationResult
from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Features_Array,
    T_Batch_Label_Array,
    T_Batch_Token_Array
)
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.io import copy_file

from sciencebeam_trainer_delft.sequence_labelling.config import TrainingConfig
from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_template import (
    get_wapiti_template_feature_indices
)
from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti import (
    WapitiModel,
    WapitiWrapper,
    format_feature_line
)


LOGGER = logging.getLogger(__name__)


def translate_tags_IOB_to_grobid(tag: str) -> str:
    """
    Convert labels from IOB2 to the ones used by GROBID (expected by the wapiti model)
    """
    if tag == 'O':
        # outside
        return '<other>'
    elif tag.startswith('B-'):
        # begin
        return 'I-' + tag[2:]
    elif tag.startswith('I-'):
        # inside
        return '' + tag[2:]
    else:
        return tag


def iter_doc_formatted_input_data(
    x_doc: np.ndarray,
    features_doc: np.ndarray
) -> Iterable[str]:
    for x_token, f_token in zip(x_doc, features_doc):
        try:
            yield format_feature_line([x_token] + list(f_token))
        except TypeError as error:
            raise RuntimeError(
                'failed to concatenate: x=<%s>, f=<%s>' % (x_token, f_token)
            ) from error
    # blank lines to mark the end of the document
    yield ''
    yield ''


def iter_formatted_input_data(
    x: np.ndarray,
    features: np.ndarray
) -> Iterable[str]:
    return (
        line + '\n'
        for x_doc, f_doc in zip(x, features)
        for line in iter_doc_formatted_input_data(x_doc, f_doc)
    )


def write_wapiti_input_data(fp: IO, x: np.ndarray, features: np.ndarray):
    fp.writelines(iter_formatted_input_data(
        x, features
    ))


def iter_read_tagged_result(fp: IO) -> Iterable[List[Tuple[str, str]]]:
    token_and_label_pairs: List[Tuple[str, str]] = []
    for line in fp:
        LOGGER.debug('line: %r', line)
        line = line.rstrip()
        if not line:
            if token_and_label_pairs:
                yield token_and_label_pairs
            token_and_label_pairs = []
            continue
        values = line.replace('\t', ' ').split(' ')
        if len(values) < 2:
            raise ValueError('should have multiple values, but got: [%s]' % line)
        token_and_label_pairs.append((
            values[0],
            translate_tags_grobid_to_IOB(values[-1])
        ))

    if token_and_label_pairs:
        yield token_and_label_pairs


def convert_wapiti_model_result_to_document_tagged_result(
        x_doc: List[str],
        wapiti_model_result: List[List[str]]) -> List[Tuple[str, str]]:
    return [
        (
            x_token,
            translate_tags_grobid_to_IOB(result_token[-1])
        )
        for x_token, result_token in zip(x_doc, wapiti_model_result)
    ]


class WapitiModelAdapter:
    def __init__(
        self,
        wapiti_wrapper: WapitiWrapper,
        model_file_path: str,
        model_path: Optional[str] = None
    ):
        self.wapiti_wrapper = wapiti_wrapper
        self.model_file_path = model_file_path
        self.model_path = model_path
        self._wapiti_model: Optional[WapitiModel] = None

    def get_read_feature_indices(self) -> Optional[Set[int]]:
        # a trained model carries its template inside the model file, where this cannot read it,
        # so every feature is treated as read
        return None

    @property
    def wapiti_model(self) -> WapitiModel:
        if self._wapiti_model is not None:
            return self._wapiti_model
        wapiti_model = self.wapiti_wrapper.load_model(self.model_file_path)
        self._wapiti_model = wapiti_model
        return wapiti_model

    @staticmethod
    def load_from(
        model_path: str,
        download_manager: DownloadManager,
        wapiti_binary_path: Optional[str] = None
    ) -> 'WapitiModelAdapter':
        model_file_path = os.path.join(model_path, 'model.wapiti.gz')
        local_model_file_path = None
        try:
            local_model_file_path = download_manager.download_if_url(model_file_path)
        except FileNotFoundError:
            pass
        if not local_model_file_path or not os.path.isfile(str(local_model_file_path)):
            model_file_path = os.path.splitext(model_file_path)[0]
            local_model_file_path = download_manager.download_if_url(model_file_path)
        LOGGER.debug('local_model_file_path: %s', local_model_file_path)
        if local_model_file_path.endswith('.gz'):
            local_uncompressed_file_path = os.path.splitext(local_model_file_path)[0]
            copy_file(local_model_file_path, local_uncompressed_file_path, overwrite=False)
            local_model_file_path = local_uncompressed_file_path
        return WapitiModelAdapter(
            WapitiWrapper(
                wapiti_binary_path=wapiti_binary_path
            ),
            model_file_path=local_model_file_path,
            model_path=model_path
        )

    def _get_model_name(self) -> str:
        return os.path.basename(os.path.dirname(self.model_file_path))

    def iter_tag_using_model(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> Iterable[List[Tuple[str, str]]]:
        # Note: this method doesn't currently seem to work reliable and needs to be investigated
        #   The evaluation always shows zero.
        assert not output_format, 'output_format not supported'
        for x_doc, f_doc in zip(x, features):
            LOGGER.debug('x_doc=%s, f_doc=%s', x_doc, f_doc)
            result = self.wapiti_model.label_features([
                [x_token] + list(f_token)
                for x_token, f_token in zip(x_doc, f_doc)
            ])
            yield convert_wapiti_model_result_to_document_tagged_result(
                x_doc,
                result
            )

    def iter_tag_using_wrapper(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> Iterable[List[Tuple[str, str]]]:
        assert not output_format, 'output_format not supported'
        with tempfile.TemporaryDirectory(suffix='wapiti') as temp_dir:
            data_path = Path(temp_dir).joinpath('input.data')
            output_data_path = Path(temp_dir).joinpath('output.data')
            with data_path.open(mode='w', encoding='utf-8') as fp:
                write_wapiti_input_data(
                    fp, x=x, features=features
                )
            self.wapiti_wrapper.label(
                model_path=self.model_file_path,
                data_path=str(data_path),
                output_data_path=str(output_data_path),
                output_only_labels=False
            )
            with output_data_path.open(mode='r', encoding='utf-8') as output_data_fp:
                yield from iter_read_tagged_result(output_data_fp)

    def iter_tag(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> Iterable[List[Tuple[str, str]]]:
        return self.iter_tag_using_wrapper(x, features, output_format)

    def tag(
        self,
        x: np.ndarray,
        features: np.ndarray,
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        assert not output_format, 'output_format not supported'
        return list(self.iter_tag(x, features))

    def eval(self, x_test, y_test, features: T_Batch_Features_Array):
        self.eval_single(x_test, y_test, features=features)

    @property
    def model_summary_props(self) -> dict:
        return {
            'model_type': 'wapiti'
        }

    def get_evaluation_result(
        self,
        x_test: T_Batch_Token_Array,
        y_test: T_Batch_Label_Array,
        features: T_Batch_Features_Array
    ) -> ClassificationResult:
        tag_result = self.tag(x_test, features)
        y_true = [
            y_token
            for y_doc in y_test
            for y_token in y_doc
        ]
        y_pred = [
            tag_result_token[-1]
            for tag_result_doc in tag_result
            for tag_result_token in tag_result_doc
        ]
        return ClassificationResult(
            y_pred=y_pred,
            y_true=y_true
        )

    def eval_single(
        self,
        x_test: T_Batch_Token_Array,
        y_test: T_Batch_Label_Array,
        features: T_Batch_Features_Array
    ):
        classification_result = self.get_evaluation_result(
            x_test=x_test,
            y_test=y_test,
            features=features
        )
        print(classification_result.get_formatted_report(digits=4))


def iter_doc_formatted_training_data(
    x_doc: np.ndarray,
    y_doc: np.ndarray,
    features_doc: np.ndarray
) -> Iterable[str]:
    for x_token, y_token, f_token in zip(x_doc, y_doc, features_doc):
        # list(), because adding a list to a numpy row concatenates the strings elementwise
        # rather than the sequences, which silently writes one mangled column per feature
        yield format_feature_line(
            [x_token] + list(f_token) + [translate_tags_IOB_to_grobid(y_token)]
        )
    # blank lines to mark the end of the document
    yield ''
    yield ''


def iter_formatted_training_data(
    x: np.ndarray,
    y: np.ndarray,
    features: np.ndarray
) -> Iterable[str]:
    return (
        line + '\n'
        for x_doc, y_doc, f_doc in zip(x, y, features)
        for line in iter_doc_formatted_training_data(x_doc, y_doc, f_doc)
    )


def write_wapiti_train_data(fp: IO, x: np.ndarray, y: np.ndarray, features: np.ndarray):
    fp.writelines(iter_formatted_training_data(
        x, y, features
    ))


class WapitiModelTrainAdapter:
    def __init__(
        self,
        model_name: str,
        template_path: str,
        temp_model_path: str,
        max_epoch: int,
        download_manager: DownloadManager,
        gzip_enabled: bool = False,
        wapiti_binary_path: Optional[str] = None,
        wapiti_train_args: Optional[dict] = None
    ):
        self.model_name = model_name
        self.template_path = template_path
        self.temp_model_path = temp_model_path
        self.max_epoch = max_epoch
        self.download_manager = download_manager
        self.gzip_enabled = gzip_enabled
        self.wapiti_binary_path = wapiti_binary_path
        self.wapiti_train_args = wapiti_train_args
        self._model_adapter: Optional[WapitiModelAdapter] = None
        self._local_template_path: Optional[str] = None
        # additional properties to keep "compatibility" with wrapper.Sequence
        self.log_dir = None
        self.model_path = None
        self.training_config = TrainingConfig(initial_epoch=0)

    def get_local_template_path(self) -> str:
        if self._local_template_path is None:
            self._local_template_path = self.download_manager.download_if_url(self.template_path)
            LOGGER.info('local_template_path: %s', self._local_template_path)
        return self._local_template_path

    def get_read_feature_indices(self) -> Optional[Set[int]]:
        # resolved before the training data is checked against it, rather than lazily in `train`
        return get_wapiti_template_feature_indices(self.get_local_template_path())

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: Optional[np.ndarray],
        y_valid: Optional[np.ndarray],
        features_train: T_Batch_Features_Array,
        features_valid: Optional[T_Batch_Features_Array]
    ):
        local_template_path = self.get_local_template_path()
        if not self.temp_model_path:
            self.temp_model_path = '/tmp/model.wapiti'
        with tempfile.TemporaryDirectory(suffix='wapiti') as temp_dir:
            data_path = Path(temp_dir).joinpath('train.data')
            with data_path.open(mode='w', encoding='utf-8') as fp:
                write_wapiti_train_data(
                    fp, x=x_train, y=y_train, features=features_train
                )
                if x_valid is not None:
                    assert y_valid is not None
                    assert features_valid is not None
                    write_wapiti_train_data(
                        fp, x=x_valid, y=y_valid, features=features_valid
                    )
            WapitiWrapper(wapiti_binary_path=self.wapiti_binary_path).train(
                data_path=str(data_path),
                output_model_path=self.temp_model_path,
                template_path=local_template_path,
                max_iter=self.max_epoch,
                **(self.wapiti_train_args or {})
            )
            LOGGER.info('wapiti model trained: %s', self.temp_model_path)

    def get_model_adapter(self) -> WapitiModelAdapter:
        if self._model_adapter is not None:
            return self._model_adapter
        assert self.temp_model_path, "temp_model_path required"
        model_adapter = WapitiModelAdapter.load_from(
            os.path.dirname(self.temp_model_path),
            download_manager=self.download_manager,
            wapiti_binary_path=self.wapiti_binary_path
        )
        self._model_adapter = model_adapter
        return model_adapter

    @property
    def last_checkpoint_path(self) -> Optional[str]:
        return None

    @property
    def model_summary_props(self) -> dict:
        return self.get_model_adapter().model_summary_props

    def get_evaluation_result(
        self,
        x_test: T_Batch_Token_Array,
        y_test: T_Batch_Label_Array,
        features: T_Batch_Features_Array
    ) -> ClassificationResult:
        return self.get_model_adapter().get_evaluation_result(
            x_test, y_test, features=features
        )

    def eval(
        self,
        x_test: List[List[str]],
        y_test: List[List[str]],
        features: T_Batch_Features_Array
    ):
        self.get_model_adapter().eval(
            x_test, y_test, features=features
        )

    def get_model_output_path(self, output_path: Optional[str] = None) -> str:
        assert output_path, "output_path required"
        return os.path.join(output_path, self.model_name)

    def save(self, output_path: Optional[str] = None):
        model_output_path = self.get_model_output_path(output_path)
        assert self.temp_model_path, "temp_model_path required"
        if not Path(self.temp_model_path).exists():
            raise FileNotFoundError("temp_model_path does not exist: %s" % self.temp_model_path)
        model_file_path = os.path.join(model_output_path, 'model.wapiti')
        if self.gzip_enabled:
            model_file_path += '.gz'
        LOGGER.info('saving to %s', model_file_path)
        copy_file(self.temp_model_path, model_file_path)
