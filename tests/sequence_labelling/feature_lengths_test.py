from types import SimpleNamespace
from typing import List, Optional, Sequence

import pytest

from sciencebeam_trainer_delft.sequence_labelling.feature_lengths import (
    FeatureLengthModes,
    InconsistentFeatureLengthError,
    InsufficientFeatureLengthError,
    check_feature_lengths,
    get_document_length,
    get_document_width,
    get_feature_length_report,
    get_model_config_read_feature_indices,
    get_read_feature_indices
)


INPUT_PATH_1 = '/path/to/input1.train'
INPUT_PATH_2 = '/path/to/input2.train'


def get_document(width: int, row_count: int = 2) -> List[List[str]]:
    return [
        ['f%d' % index for index in range(width)]
        for _ in range(row_count)
    ]


def get_documents(widths: Sequence[int]) -> List[List[List[str]]]:
    return [get_document(width) for width in widths]


def get_report(*widths_by_input: Sequence[int]):
    input_paths = [INPUT_PATH_1, INPUT_PATH_2][:len(widths_by_input)]
    assert len(input_paths) == len(widths_by_input)
    return get_feature_length_report(
        input_paths=input_paths,
        features_list=[get_documents(widths) for widths in widths_by_input]
    )


def get_read_indices(*indices: int) -> Optional[set]:
    return set(indices)


class TestGetDocumentLength:
    def test_should_return_min_and_max_row_length(self):
        assert get_document_length([['f1', 'f2'], ['f1', 'f2', 'f3']]) == (2, 3)

    def test_should_return_none_for_document_without_rows(self):
        assert get_document_length([]) is None

    def test_should_return_width_for_uniform_document(self):
        assert get_document_width(get_document_length(get_document(3))) == 3

    def test_should_return_no_width_for_document_with_mixed_rows(self):
        assert get_document_width(get_document_length([['f1'], ['f1', 'f2']])) is None


class TestFeatureLengthReport:
    def test_should_count_documents_by_width_per_input(self):
        report = get_report([30, 30, 31], [31])
        assert report.input_lengths[0].width_counts == {30: 2, 31: 1}
        assert report.input_lengths[1].width_counts == {31: 1}
        assert report.width_counts == {30: 2, 31: 2}
        assert report.document_count == 4

    def test_should_be_uniform_for_single_width(self):
        assert get_report([30, 30], [30]).is_uniform

    def test_should_not_be_uniform_where_a_document_has_mixed_rows(self):
        report = get_feature_length_report(
            input_paths=[INPUT_PATH_1],
            features_list=[[get_document(30), [['f1'], ['f1', 'f2']]]]
        )
        assert not report.is_uniform
        assert report.width_counts == {30: 1, None: 1}

    def test_should_name_each_input_in_the_table(self):
        table = get_report([30], [31]).format_table()
        assert INPUT_PATH_1 in table
        assert INPUT_PATH_2 in table
        assert '30' in table
        assert '31' in table

    def test_should_choose_the_majority_width(self):
        assert get_report([30, 30, 30], [31, 31]).get_majority_width() == 30

    def test_should_break_majority_ties_by_input_order(self):
        assert get_report([31], [30]).get_majority_width() == 31
        assert get_report([30], [31]).get_majority_width() == 30

    def test_should_not_let_a_mixed_document_set_the_majority(self):
        report = get_feature_length_report(
            input_paths=[INPUT_PATH_1],
            features_list=[[
                [['f1'], ['f1', 'f2']], [['f1'], ['f1', 'f2']], get_document(30)
            ]]
        )
        assert report.get_majority_width() == 30


class TestCheckFeatureLengthsFail:
    def test_should_accept_a_uniform_corpus(self):
        result = check_feature_lengths(get_report([30, 30], [30]), get_read_indices(9, 29))
        assert result.masks is None

    def test_should_raise_on_differing_widths(self):
        with pytest.raises(InconsistentFeatureLengthError) as exc_info:
            check_feature_lengths(get_report([30, 30], [31]), get_read_indices(9, 10))
        message = str(exc_info.value)
        assert '30' in message and '31' in message
        assert '--on-inconsistent-feature-lengths=accept' in message
        assert '--on-inconsistent-feature-lengths=drop' in message

    def test_should_raise_on_differing_widths_within_a_single_input(self):
        with pytest.raises(InconsistentFeatureLengthError):
            check_feature_lengths(get_report([30, 31]), get_read_indices(9))

    def test_should_raise_where_a_read_index_is_out_of_range_for_a_uniform_corpus(self):
        with pytest.raises(InsufficientFeatureLengthError) as exc_info:
            check_feature_lengths(get_report([30, 30]), get_read_indices(9, 31))
        message = str(exc_info.value)
        assert '32' in message
        assert '2 of 2 documents' in message

    def test_should_not_check_sufficiency_where_all_features_are_read(self):
        assert check_feature_lengths(get_report([30, 30]), None).masks is None


class TestCheckFeatureLengthsAccept:
    def test_should_proceed_where_every_read_index_is_present(self):
        result = check_feature_lengths(
            get_report([30, 30], [31]),
            get_read_indices(9, 29),
            mode=FeatureLengthModes.ACCEPT
        )
        assert result.masks is None
        assert result.required_feature_count == 30

    def test_should_raise_where_a_read_index_is_missing_from_some_documents(self):
        with pytest.raises(InsufficientFeatureLengthError):
            check_feature_lengths(
                get_report([30, 30], [31]),
                get_read_indices(9, 30),
                mode=FeatureLengthModes.ACCEPT
            )

    def test_should_raise_where_the_run_reads_every_feature(self):
        with pytest.raises(InconsistentFeatureLengthError) as exc_info:
            check_feature_lengths(
                get_report([30, 30], [31]),
                None,
                mode=FeatureLengthModes.ACCEPT
            )
        assert '--features-indices' in str(exc_info.value)

    def test_should_not_raise_for_a_uniform_corpus_reading_every_feature(self):
        assert check_feature_lengths(
            get_report([30, 30]), None, mode=FeatureLengthModes.ACCEPT
        ).masks is None


class TestCheckFeatureLengthsDrop:
    def test_should_keep_every_document_where_the_read_indices_are_present(self):
        result = check_feature_lengths(
            get_report([30, 30], [31]),
            get_read_indices(9, 29),
            mode=FeatureLengthModes.DROP
        )
        assert result.masks is None
        assert result.dropped_document_count == 0

    def test_should_drop_documents_missing_a_read_index(self):
        result = check_feature_lengths(
            get_report([30, 30], [31]),
            get_read_indices(9, 30),
            mode=FeatureLengthModes.DROP
        )
        assert result.masks == [[False, False], [True]]
        assert result.dropped_document_count == 2

    def test_should_read_up_to_the_majority_width_where_every_feature_is_read(self):
        result = check_feature_lengths(
            get_report([30, 30, 30], [31, 31]),
            None,
            mode=FeatureLengthModes.DROP
        )
        assert result.required_feature_count == 30
        # the 31-column documents supply the first 30 features, so nothing is dropped
        assert result.masks is None

    def test_should_drop_narrower_documents_where_the_majority_is_wider(self):
        result = check_feature_lengths(
            get_report([31, 31, 31], [30]),
            None,
            mode=FeatureLengthModes.DROP
        )
        assert result.required_feature_count == 31
        assert result.masks == [[True, True, True], [False]]

    def test_should_raise_where_dropping_would_leave_nothing(self):
        with pytest.raises(InsufficientFeatureLengthError):
            check_feature_lengths(
                get_report([30, 30]),
                get_read_indices(31),
                mode=FeatureLengthModes.DROP
            )


class TestCheckFeatureLengthsIgnore:
    def test_should_never_raise_or_drop(self):
        result = check_feature_lengths(
            get_report([30, 30], [31]),
            get_read_indices(31),
            mode=FeatureLengthModes.IGNORE
        )
        assert result.masks is None


class TestCheckFeatureLengthsExpectedFeatureCount:
    def test_should_raise_where_a_uniform_corpus_does_not_match_what_the_run_settled_on(self):
        with pytest.raises(InconsistentFeatureLengthError) as exc_info:
            check_feature_lengths(
                get_report([31, 31]),
                get_read_indices(9),
                expected_feature_count=30
            )
        message = str(exc_info.value)
        assert '31' in message and '30' in message

    def test_should_not_raise_where_the_corpus_matches(self):
        assert check_feature_lengths(
            get_report([30, 30]),
            get_read_indices(9),
            expected_feature_count=30
        ).masks is None

    def test_should_accept_a_wider_corpus_where_every_read_feature_is_present(self):
        assert check_feature_lengths(
            get_report([31, 31]),
            get_read_indices(9),
            mode=FeatureLengthModes.ACCEPT,
            expected_feature_count=30
        ).masks is None

    def test_should_read_up_to_the_adopted_count_rather_than_the_local_majority(self):
        # every feature is read, so without an adopted count this corpus would read 31
        result = check_feature_lengths(
            get_report([31, 31, 31]),
            None,
            mode=FeatureLengthModes.DROP,
            expected_feature_count=30
        )
        assert result.required_feature_count == 30
        assert result.masks is None

    def test_should_drop_documents_narrower_than_the_adopted_count(self):
        result = check_feature_lengths(
            get_report([30, 29]),
            None,
            mode=FeatureLengthModes.DROP,
            expected_feature_count=30
        )
        assert result.masks == [[True, False]]

    def test_should_raise_where_dropping_against_the_adopted_count_leaves_nothing(self):
        with pytest.raises(InsufficientFeatureLengthError):
            check_feature_lengths(
                get_report([29, 29]),
                None,
                mode=FeatureLengthModes.DROP,
                expected_feature_count=30
            )

    def test_should_report_the_adopted_count_of_a_uniform_corpus(self):
        assert check_feature_lengths(
            get_report([30, 30]), get_read_indices(9)
        ).adopted_feature_count == 30

    def test_should_report_no_adopted_count_for_accepted_mixed_widths(self):
        assert check_feature_lengths(
            get_report([30], [31]), get_read_indices(9), mode=FeatureLengthModes.ACCEPT
        ).adopted_feature_count is None


class TestGetReadFeatureIndices:
    def test_should_read_every_feature_where_features_are_used_without_a_selection(self):
        assert get_read_feature_indices(use_features=True) is None

    def test_should_read_the_selected_features(self):
        assert get_read_feature_indices(
            use_features=True, features_indices=[9, 10]
        ) == {9, 10}

    def test_should_read_continuous_features_alongside_the_selected_ones(self):
        assert get_read_feature_indices(
            use_features=True, features_indices=[9, 10], continuous_features_indices=[22]
        ) == {9, 10, 22}

    def test_should_read_no_features_where_none_are_used(self):
        assert get_read_feature_indices(use_features=False) == set()

    def test_should_read_token_and_text_features_even_where_features_are_not_used(self):
        assert get_read_feature_indices(
            use_features=False,
            additional_token_feature_indices=[0],
            text_feature_indices=[32]
        ) == {0, 32}

    def test_should_read_the_unrolled_text_feature(self):
        assert get_read_feature_indices(unroll_text_feature_index=33) == {33}

    def test_should_read_from_a_model_config(self):
        model_config = SimpleNamespace(
            use_features=True,
            features_indices=[9, 10],
            continuous_features_indices=None,
            additional_token_feature_indices=None,
            text_feature_indices=None,
            unroll_text_feature_index=None
        )
        assert get_model_config_read_feature_indices(model_config) == {9, 10}

    def test_should_tolerate_a_model_config_without_the_newer_keys(self):
        assert get_model_config_read_feature_indices(
            SimpleNamespace(use_features=True, features_indices=[9])
        ) == {9}
