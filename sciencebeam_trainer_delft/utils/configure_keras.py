# see https://github.com/keras-team/keras/issues/1406

from contextlib import redirect_stderr
import os

# Use tf_keras (Keras 2) via tf.keras to ensure compatibility with tensorflow_addons
# which uses tf.keras directly. Without this, Keras 3 is used and causes errors.
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

with redirect_stderr(open(os.devnull, "w", encoding='utf-8')):
    import tf_keras  # noqa pylint: disable=unused-import
