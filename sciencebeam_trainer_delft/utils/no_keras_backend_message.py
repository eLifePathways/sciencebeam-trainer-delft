# see https://github.com/keras-team/keras/issues/1406

from contextlib import redirect_stderr
import os

with redirect_stderr(open(os.devnull, "w", encoding='utf-8')):
    import tf_keras  # noqa pylint: disable=unused-import
