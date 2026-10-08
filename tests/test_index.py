import json

import numpy as np
import pytest

from app.index import Index, IndexLoadError


def test_save_and_load_roundtrip(index, tmp_path):
    folder = str(tmp_path / "idx")
    index.save(folder)
    loaded = Index.load(folder, "hash")
    assert len(loaded) == len(index)
    assert loaded.chunks[0] == index.chunks[0]
    assert np.allclose(loaded.embeddings, index.embeddings)
    assert loaded.meta["embedding_model"] == "hash"


def test_saving_twice_replaces_old_index(index, tmp_path):
    folder = str(tmp_path / "idx")
    index.save(folder)
    index.save(folder)
    assert not (tmp_path / "idx.new").exists()
    assert not (tmp_path / "idx.old").exists()
    assert Index.load(folder, "hash").meta["n_chunks"] == len(index)


def test_load_refuses_a_different_embedding_model(index, tmp_path):
    folder = str(tmp_path / "idx")
    index.save(folder)
    with pytest.raises(IndexLoadError, match="Rebuild the index"):
        Index.load(folder, "some-other-model")


def test_load_missing_index(tmp_path):
    with pytest.raises(IndexLoadError, match="scripts.ingest"):
        Index.load(str(tmp_path / "nothing"), "hash")


def test_mismatched_files_are_rejected(index, tmp_path):
    folder = tmp_path / "idx"
    index.save(str(folder))
    np.save(folder / "embeddings.npy", index.embeddings[:-1])
    with pytest.raises(IndexLoadError):
        Index.load(str(folder), "hash")


def test_meta_describes_the_index(index):
    assert index.meta["n_chunks"] == len(index)
    assert "warfarin" in index.meta["books"]
    json.dumps(index.meta)  # must be serializable
