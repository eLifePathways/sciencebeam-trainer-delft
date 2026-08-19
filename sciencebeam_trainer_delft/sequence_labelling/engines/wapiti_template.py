import logging
import re
from typing import Iterable, Optional, Set


LOGGER = logging.getLogger(__name__)


# wapiti patterns are `%x[row,col]`, with `%t`/`%m` variants and an optional trailing marker;
# a single line may hold several, joined by `/`
WAPITI_PATTERN_COLUMN_PATTERN = re.compile(r'%[xXtTmM]\[\s*[+-]?\d+\s*,\s*(\d+)')

# column 0 of the data written for wapiti is the token itself, features follow it
TOKEN_COLUMN_COUNT = 1


def iter_template_columns(lines: Iterable[str]) -> Iterable[int]:
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            # a commented-out pattern is not applied, so it reads nothing
            continue
        for match in WAPITI_PATTERN_COLUMN_PATTERN.finditer(stripped_line):
            yield int(match.group(1))


def get_template_feature_indices(lines: Iterable[str]) -> Optional[Set[int]]:
    """The feature indices a template reads, or None where no pattern could be parsed.

    Feature index `i` is column `i + 1`, because the training data written for wapiti is
    `[token] + features + [label]`. A template referencing only column 0 reads the token
    and no features, which is an empty set rather than an unknown one.
    """
    columns = set(iter_template_columns(lines))
    if not columns:
        return None
    return {
        column - TOKEN_COLUMN_COUNT
        for column in columns
        if column >= TOKEN_COLUMN_COUNT
    }


def get_wapiti_template_feature_indices(template_path: str) -> Optional[Set[int]]:
    with open(template_path, 'r', encoding='utf-8') as fp:
        feature_indices = get_template_feature_indices(fp)
    if feature_indices is None:
        LOGGER.warning(
            'no wapiti pattern found in template (%r); treating every feature as read',
            template_path
        )
    else:
        LOGGER.info(
            'wapiti template reads %d features, up to index %s (%r)',
            len(feature_indices),
            max(feature_indices) if feature_indices else None,
            template_path
        )
    return feature_indices
