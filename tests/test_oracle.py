import json

import pytest

from eval.corpus import OracleProvider, StaticLocalProvider
from oracle.oracle import write_corpus


def _make_corpus(tmp_path):
    docs = [
        ("Article One", "2026-06-16T00:00:00Z", "alpha beta gamma " * 200),
        ("Article Two", "2026-06-16T00:01:00Z", "delta epsilon zeta " * 200),
    ]
    write_corpus(tmp_path, docs, chunk_bytes=1024)
    return tmp_path


def test_oracle_corpus_excludes_metadata(tmp_path):
    corpus = _make_corpus(tmp_path)
    # write a manifest.json + provenance.json (write_corpus already wrote provenance.json)
    provider = StaticLocalProvider(corpus)
    manifest = provider.manifest()
    (corpus / "manifest.json").write_text(json.dumps({"manifest_hash": manifest.manifest_hash()}))

    # Adding metadata files must NOT change the sampled corpus size or hash.
    reloaded = StaticLocalProvider(corpus)
    assert reloaded.total_bytes == provider.total_bytes
    assert reloaded.manifest().manifest_hash() == manifest.manifest_hash()
    assert all(c.id.startswith("chunk_") for c in reloaded.manifest().chunks)


def test_oracle_provider_hash_verification(tmp_path):
    corpus = _make_corpus(tmp_path)
    good_hash = StaticLocalProvider(corpus).manifest().manifest_hash()

    provider = OracleProvider(corpus, expected_manifest_hash=good_hash)
    assert provider.total_bytes > 0

    with pytest.raises(ValueError):
        OracleProvider(corpus, expected_manifest_hash="deadbeef")
