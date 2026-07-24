from tools.record_visibility import hide_record, load_hidden_record_ids


def test_hidden_record_state_is_category_scoped_and_idempotent(tmp_path):
    data_root = tmp_path / "data"

    hide_record(data_root, "model_update_jobs", "job-1")
    hide_record(data_root, "model_update_jobs", "job-1")
    hide_record(data_root, "submission_history", "submission-1")

    assert load_hidden_record_ids(data_root, "model_update_jobs") == {"job-1"}
    assert load_hidden_record_ids(data_root, "submission_history") == {
        "submission-1"
    }
