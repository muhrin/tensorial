import pytest

from tensorial.gcnn.utils import UpdateDict


def test_updatedict_getitem():
    d = {"a": 1, "b": {"c": 2}}
    u = UpdateDict(d)

    assert u["a"] == 1
    assert isinstance(u["b"], UpdateDict)
    assert u["b"]["c"] == 2

    # KeyError for missing
    with pytest.raises(KeyError):
        _ = u["x"]


def test_updatedict_setitem():
    d = {"a": 1}
    u = UpdateDict(d)

    u["a"] = 10
    u["b"] = 20

    assert u["a"] == 10
    assert u["b"] == 20
    assert d["a"] == 1  # Original untouched
    assert "b" not in d


def test_updatedict_delitem():
    d = {"a": 1, "b": 2}
    u = UpdateDict(d)

    del u["a"]

    with pytest.raises(KeyError):
        _ = u["a"]

    u["c"] = 3
    del u["c"]

    with pytest.raises(KeyError):
        _ = u["c"]


def test_updatedict_iter():
    d = {"a": 1, "b": 2}
    u = UpdateDict(d)

    u["c"] = 3
    del u["b"]

    keys = set(u)
    assert keys == {"a", "c"}


def test_updatedict_len():
    d = {"a": 1, "b": 2}
    u = UpdateDict(d)

    assert len(u) == 2
    u["c"] = 3
    assert len(u) == 3
    del u["a"]
    assert len(u) == 2


def test_updatedict_asdict():
    d = {"a": 1, "b": {"c": 2}, "d": 4}
    u = UpdateDict(d)

    u["a"] = 10
    u["b"]["c"] = 20
    u["b"]["x"] = 30
    u["e"] = 50
    del u["d"]

    result = u._asdict()
    assert result == {"a": 10, "b": {"c": 20, "x": 30}, "e": 50}
    # original untouched
    assert d == {"a": 1, "b": {"c": 2}, "d": 4}
