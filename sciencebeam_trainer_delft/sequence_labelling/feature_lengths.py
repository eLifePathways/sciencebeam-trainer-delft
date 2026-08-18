import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOGGER = logging.getLogger(__name__)


class FeatureLengthModes:
    FAIL = 'fail'
    ACCEPT = 'accept'
    DROP = 'drop'
    # not selectable from the command line: `input_info` reports and changes nothing
    IGNORE = 'ignore'


# the modes a caller may select; IGNORE is deliberately absent
FEATURE_LENGTH_MODES = [
    FeatureLengthModes.FAIL,
    FeatureLengthModes.ACCEPT,
    FeatureLengthModes.DROP
]

DEFAULT_FEATURE_LENGTH_MODE = FeatureLengthModes.FAIL

FEATURE_LENGTH_MODE_ARG_NAME = '--on-inconsistent-feature-lengths'


class InconsistentFeatureLengthError(ValueError):
    pass


class InsufficientFeatureLengthError(ValueError):
    pass


# a document holds one (min, max) row length pair, or None where it has no rows
T_Document_Length = Optional[Tuple[int, int]]


def get_document_length(features_doc: Iterable[Sequence[str]]) -> T_Document_Length:
    row_lengths = [len(features_vector) for features_vector in features_doc]
    if not row_lengths:
        return None
    return (min(row_lengths), max(row_lengths))


def get_document_lengths(
    features_all: Iterable[Iterable[Sequence[str]]]
) -> List[T_Document_Length]:
    return [get_document_length(features_doc) for features_doc in features_all]


def get_document_width(document_length: T_Document_Length) -> Optional[int]:
    """The single feature length of a document, or None where its rows disagree."""
    if document_length is None:
        return None
    min_length, max_length = document_length
    if min_length != max_length:
        return None
    return min_length


def is_document_length_sufficient(
    document_length: T_Document_Length,
    required_feature_count: int
) -> bool:
    if document_length is None:
        return True
    return document_length[0] >= required_feature_count


@dataclass
class InputFeatureLengths:
    """The feature lengths of the documents loaded from one input."""
    input_path: str
    document_lengths: List[T_Document_Length] = field(default_factory=list)

    @property
    def document_count(self) -> int:
        return len(self.document_lengths)

    @property
    def width_counts(self) -> Dict[Optional[int], int]:
        """Document counts by feature length; the None key counts documents whose rows disagree."""
        counts: Counter = Counter()
        for document_length in self.document_lengths:
            if document_length is None:
                continue
            counts[get_document_width(document_length)] += 1
        return counts

    def get_insufficient_document_count(self, required_feature_count: int) -> int:
        return sum(
            1
            for document_length in self.document_lengths
            if not is_document_length_sufficient(document_length, required_feature_count)
        )

    def get_sufficient_document_mask(self, required_feature_count: int) -> List[bool]:
        return [
            is_document_length_sufficient(document_length, required_feature_count)
            for document_length in self.document_lengths
        ]

    def format_widths(self) -> str:
        return format_width_counts(self.width_counts)


def format_document_count(count: int) -> str:
    return '%d document%s' % (count, '' if count == 1 else 's')


def format_width_counts(width_counts: Dict[Optional[int], int]) -> str:
    if not width_counts:
        return 'none'
    return ', '.join(
        '%s: %d documents' % (
            'mixed within document' if width is None else width,
            count
        )
        for width, count in sorted(
            width_counts.items(),
            key=lambda item: (item[0] is None, item[0])
        )
    )


@dataclass
class FeatureLengthReport:
    """The feature lengths of every input of one run, before they are concatenated."""
    input_lengths: List[InputFeatureLengths] = field(default_factory=list)

    @property
    def document_count(self) -> int:
        return sum(item.document_count for item in self.input_lengths)

    @property
    def width_counts(self) -> Dict[Optional[int], int]:
        counts: Counter = Counter()
        for item in self.input_lengths:
            counts.update(item.width_counts)
        return counts

    @property
    def widths_in_input_order(self) -> List[Optional[int]]:
        """Distinct feature lengths, in the order the inputs first present them."""
        widths: List[Optional[int]] = []
        for item in self.input_lengths:
            for document_length in item.document_lengths:
                width = get_document_width(document_length)
                if document_length is not None and width not in widths:
                    widths.append(width)
        return widths

    @property
    def is_uniform(self) -> bool:
        width_counts = self.width_counts
        return len(width_counts) <= 1 and None not in width_counts

    @property
    def min_feature_count(self) -> Optional[int]:
        min_lengths = [
            document_length[0]
            for item in self.input_lengths
            for document_length in item.document_lengths
            if document_length is not None
        ]
        if not min_lengths:
            return None
        return min(min_lengths)

    def get_majority_width(self) -> Optional[int]:
        """The feature length held by the most documents, ties broken by input order.

        Documents whose own rows disagree hold no length and cannot set it.
        """
        width_counts = self.width_counts
        candidates = [
            width for width in self.widths_in_input_order if width is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda width: (width_counts[width], -candidates.index(width)))

    def get_insufficient_document_count(self, required_feature_count: int) -> int:
        return sum(
            item.get_insufficient_document_count(required_feature_count)
            for item in self.input_lengths
        )

    @property
    def uniform_feature_count(self) -> Optional[int]:
        """The one feature length every document holds, where there is one."""
        width_counts = self.width_counts
        if len(width_counts) != 1 or None in width_counts:
            return None
        return next(iter(width_counts))

    def get_input_paths_by_width(self, width: Optional[int]) -> List[str]:
        return [
            item.input_path
            for item in self.input_lengths
            if width in item.width_counts
        ]

    def format_table(self) -> str:
        return '\n'.join(
            '  %s: %s' % (item.input_path, item.format_widths())
            for item in self.input_lengths
        )

    def format_summary(self) -> str:
        """One line naming each length, the documents holding it and the inputs it came from."""
        width_counts = self.width_counts
        if not width_counts:
            return 'none'
        return ', '.join(
            '%s: %s (%s)' % (
                'mixed within document' if width is None else width,
                format_document_count(count),
                ', '.join(self.get_input_paths_by_width(width))
            )
            for width, count in sorted(
                width_counts.items(),
                key=lambda item: (item[0] is None, item[0])
            )
        )


def get_feature_length_report(
    input_paths: Sequence[str],
    features_list: Sequence[Iterable[Iterable[Sequence[str]]]]
) -> FeatureLengthReport:
    assert len(input_paths) == len(features_list)
    return FeatureLengthReport(input_lengths=[
        InputFeatureLengths(
            input_path=input_path,
            document_lengths=get_document_lengths(features_all)
        )
        for input_path, features_all in zip(input_paths, features_list)
    ])


def get_required_feature_count(read_feature_indices: Optional[Set[int]]) -> Optional[int]:
    """The number of features a document must hold, or None where the run reads all of them."""
    if read_feature_indices is None:
        return None
    if not read_feature_indices:
        return 0
    return max(read_feature_indices) + 1


def get_read_feature_indices(
    use_features: bool = False,
    features_indices: Optional[Iterable[int]] = None,
    continuous_features_indices: Optional[Iterable[int]] = None,
    additional_token_feature_indices: Optional[Iterable[int]] = None,
    text_feature_indices: Optional[Iterable[int]] = None,
    unroll_text_feature_index: Optional[int] = None
) -> Optional[Set[int]]:
    """The feature indices a run reads, or None where it reads every feature there is.

    Token, text and unrolled text features are read whether or not `use_features` is set;
    continuous features are read alongside the selected ones rather than from among them.
    """
    indices: Set[int] = set()
    if use_features:
        if not features_indices:
            return None
        indices.update(features_indices)
        indices.update(continuous_features_indices or [])
    indices.update(additional_token_feature_indices or [])
    indices.update(text_feature_indices or [])
    if unroll_text_feature_index is not None:
        indices.add(unroll_text_feature_index)
    return indices


def get_model_config_read_feature_indices(model_config) -> Optional[Set[int]]:
    """As `get_read_feature_indices`, for a model config; a loaded config may lack newer keys."""
    return get_read_feature_indices(
        use_features=getattr(model_config, 'use_features', False),
        features_indices=getattr(model_config, 'features_indices', None),
        continuous_features_indices=getattr(model_config, 'continuous_features_indices', None),
        additional_token_feature_indices=getattr(
            model_config, 'additional_token_feature_indices', None
        ),
        text_feature_indices=getattr(model_config, 'text_feature_indices', None),
        unroll_text_feature_index=getattr(model_config, 'unroll_text_feature_index', None)
    )


@dataclass
class FeatureLengthCheckResult:
    report: FeatureLengthReport
    required_feature_count: Optional[int] = None
    # one mask per input, or None where every document is kept
    masks: Optional[List[List[bool]]] = None
    # the feature length this run settled on, where it settled on one; a later load of the
    # same run (an evaluation corpus) is held to it
    adopted_feature_count: Optional[int] = None

    @property
    def dropped_document_count(self) -> int:
        if self.masks is None:
            return 0
        return sum(len(mask) - sum(mask) for mask in self.masks)


def _get_mode_hint(*modes: str) -> str:
    return ' or '.join(
        '%s=%s' % (FEATURE_LENGTH_MODE_ARG_NAME, mode)
        for mode in modes
    )


def _raise_inconsistent(report: FeatureLengthReport, message: str):
    LOGGER.error('inconsistent feature lengths, by input:\n%s', report.format_table())
    raise InconsistentFeatureLengthError(message)


def _raise_insufficient(report: FeatureLengthReport, message: str):
    LOGGER.error('insufficient feature lengths, by input:\n%s', report.format_table())
    raise InsufficientFeatureLengthError(message)


def _check_sufficiency(
    report: FeatureLengthReport,
    required_feature_count: int
) -> int:
    """Returns the number of documents that cannot supply the features the run reads."""
    insufficient_count = report.get_insufficient_document_count(required_feature_count)
    if insufficient_count:
        _raise_insufficient(report, (
            'the run reads %d features, but %d of %d documents hold fewer'
            ' (shortest: %s); feature lengths by input: %s; pass %s to drop those documents'
        ) % (
            required_feature_count,
            insufficient_count,
            report.document_count,
            report.min_feature_count,
            report.format_summary(),
            _get_mode_hint(FeatureLengthModes.DROP)
        ))
    return insufficient_count


def check_feature_lengths(
    report: FeatureLengthReport,
    read_feature_indices: Optional[Set[int]] = None,
    mode: str = DEFAULT_FEATURE_LENGTH_MODE,
    expected_feature_count: Optional[int] = None
) -> FeatureLengthCheckResult:
    """Applies the uniformity and sufficiency checks to what was loaded, before it is combined.

    Uniformity is what `mode` relaxes; sufficiency is never relaxed, only satisfied by dropping.

    `expected_feature_count` is a length this run has already settled on, from an earlier load
    of the same run: an evaluation corpus is held to the length its training data had, since a
    model evaluated on features it was not trained on is the same defect.
    """
    required_feature_count = get_required_feature_count(read_feature_indices)
    if required_feature_count is None and expected_feature_count is not None:
        # reading every feature of a corpus the run already fixed the width of
        required_feature_count = expected_feature_count
    if mode == FeatureLengthModes.IGNORE:
        return FeatureLengthCheckResult(
            report=report, required_feature_count=required_feature_count
        )

    if mode == FeatureLengthModes.DROP:
        # required_feature_count already carries any width adopted earlier in the run
        return _check_with_drop(report, required_feature_count)

    if not report.is_uniform:
        if mode == FeatureLengthModes.FAIL:
            _raise_inconsistent(report, (
                'inconsistent feature lengths (%s); pass %s to use them where every feature'
                ' the run reads is present in all documents, or %s to drop the rest'
            ) % (
                report.format_summary(),
                _get_mode_hint(FeatureLengthModes.ACCEPT),
                _get_mode_hint(FeatureLengthModes.DROP)
            ))
        assert mode == FeatureLengthModes.ACCEPT, 'unsupported mode: %r' % mode
        if required_feature_count is None:
            _raise_inconsistent(report, (
                'inconsistent feature lengths (%s), and the run reads every feature, so reading'
                ' all of the wider documents reads an index the narrower ones do not hold;'
                ' select the features to read (e.g. --features-indices) or pass %s'
            ) % (
                report.format_summary(),
                _get_mode_hint(FeatureLengthModes.DROP)
            ))
    elif (
        mode == FeatureLengthModes.FAIL
        and expected_feature_count is not None
        and report.uniform_feature_count != expected_feature_count
    ):
        _raise_inconsistent(report, (
            'these documents hold %s features, but this run settled on %d from the data loaded'
            ' before them (%s); pass %s to use them where every feature the run reads is present,'
            ' or %s to drop the ones that cannot supply it'
        ) % (
            report.uniform_feature_count,
            expected_feature_count,
            report.format_summary(),
            _get_mode_hint(FeatureLengthModes.ACCEPT),
            _get_mode_hint(FeatureLengthModes.DROP)
        ))

    if required_feature_count is not None:
        _check_sufficiency(report, required_feature_count)

    if not report.is_uniform:
        LOGGER.warning(
            (
                'feature lengths differ between the loaded documents (%s), accepted because every'
                ' feature the run reads (%d) is present in all of them.'
                ' Feature meaning is positional, so this assumes the surplus columns are trailing;'
                ' by input:\n%s'
            ),
            report.format_summary(),
            required_feature_count,
            report.format_table()
        )
    return FeatureLengthCheckResult(
        report=report,
        required_feature_count=required_feature_count,
        adopted_feature_count=(
            expected_feature_count
            if expected_feature_count is not None
            else report.uniform_feature_count
        )
    )


def _check_with_drop(
    report: FeatureLengthReport,
    required_feature_count: Optional[int]
) -> FeatureLengthCheckResult:
    if required_feature_count is None:
        # the run reads every feature, so the width it reads up to has to be chosen
        majority_width = report.get_majority_width()
        if majority_width is None:
            return FeatureLengthCheckResult(report=report)
        required_feature_count = majority_width
        LOGGER.warning(
            'reading %d features, the length held by the most documents (%s), by input:\n%s',
            required_feature_count,
            report.format_summary(),
            report.format_table()
        )
    masks = [
        item.get_sufficient_document_mask(required_feature_count)
        for item in report.input_lengths
    ]
    dropped_count = sum(len(mask) - sum(mask) for mask in masks)
    if not dropped_count:
        if not report.is_uniform:
            LOGGER.warning(
                'feature lengths differ between the loaded documents (%s), but every feature the'
                ' run reads (%d) is present in all of them, so none are dropped; by input:\n%s',
                report.format_summary(),
                required_feature_count,
                report.format_table()
            )
        return FeatureLengthCheckResult(
            report=report,
            required_feature_count=required_feature_count,
            adopted_feature_count=required_feature_count
        )
    document_count = report.document_count
    if dropped_count >= document_count:
        _raise_insufficient(report, (
            'every one of the %d documents holds fewer than the %d features the run reads'
            ' (shortest: %s), so dropping them leaves nothing to work with;'
            ' feature lengths by input: %s'
        ) % (
            document_count,
            required_feature_count,
            report.min_feature_count,
            report.format_summary()
        ))
    LOGGER.warning(
        (
            'dropping %d of %d documents (%.1f%%) that hold fewer than the %d features the run'
            ' reads; feature lengths by input:\n%s'
        ),
        dropped_count,
        document_count,
        100.0 * dropped_count / document_count,
        required_feature_count,
        report.format_table()
    )
    return FeatureLengthCheckResult(
        report=report,
        required_feature_count=required_feature_count,
        masks=masks,
        adopted_feature_count=required_feature_count
    )
