from sciencebeam_trainer_delft.sequence_labelling.engines.wapiti_template import (
    get_template_feature_indices,
    get_wapiti_template_feature_indices
)


class TestGetTemplateFeatureIndices:
    def test_should_map_columns_to_feature_indices(self):
        # column 0 is the token, so column 1 is feature index 0
        assert get_template_feature_indices(['U00:%x[0,1]', 'U01:%x[0,4]']) == {0, 3}

    def test_should_not_return_a_feature_index_for_the_token_column(self):
        assert get_template_feature_indices(['U00:%x[0,0]', 'U01:%x[-1,0]']) == set()

    def test_should_read_every_pattern_on_a_line(self):
        assert get_template_feature_indices(['U09:%x[-1,1]/%x[0,2]']) == {0, 1}

    def test_should_ignore_commented_out_patterns(self):
        assert get_template_feature_indices([
            'U00:%x[0,1]',
            '#U0B:%x[0,9]',
            '  # U0C:%x[0,8]'
        ]) == {0}

    def test_should_ignore_blank_lines_and_bigram_marker(self):
        assert get_template_feature_indices(['', 'U00:%x[0,3]', '', 'B']) == {2}

    def test_should_accept_negative_row_offsets(self):
        assert get_template_feature_indices(['U00:%x[-4,7]']) == {6}

    def test_should_accept_the_t_and_m_pattern_variants(self):
        assert get_template_feature_indices(['U00:%t[0,2]', 'U01:%m[0,5,"a*"]']) == {1, 4}

    def test_should_return_none_where_no_pattern_is_present(self):
        assert get_template_feature_indices(['# nothing here', 'B', '']) is None


class TestGetWapitiTemplateFeatureIndices:
    def test_should_read_the_template_from_a_file(self, tmp_path):
        template_path = tmp_path / 'template.txt'
        template_path.write_text('U00:%x[0,1]\nU01:%x[0,10]\n\nB\n', encoding='utf-8')
        assert get_wapiti_template_feature_indices(str(template_path)) == {0, 9}

    def test_should_return_none_for_a_template_without_patterns(self, tmp_path):
        template_path = tmp_path / 'template.txt'
        template_path.write_text('B\n', encoding='utf-8')
        assert get_wapiti_template_feature_indices(str(template_path)) is None
