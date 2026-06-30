from validator.github_bot import _version_for


def test_patch_within_active_series():
    assert _version_for((0, 3, 5), "0.3") == "recipe-v0.3.6"


def test_cutover_opens_new_minor_line():
    assert _version_for((0, 2, 22), "0.3") == "recipe-v0.3.0"


def test_continues_old_series_if_unbumped():
    assert _version_for((0, 2, 22), "0.2") == "recipe-v0.2.23"


def test_major_reset():
    assert _version_for((0, 9, 4), "1.0") == "recipe-v1.0.0"
