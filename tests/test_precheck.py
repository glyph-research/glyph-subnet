import json
from pathlib import Path

from validation.precheck import precheck_artifact_dir

REFERENCE_CODEC = Path(__file__).resolve().parents[1] / "reference_codec"


def test_reference_codec_passes_precheck():
    result = precheck_artifact_dir(REFERENCE_CODEC, "glyph/ref", "local")
    assert result.ok is True
    assert result.artifact_hash and len(result.artifact_hash) == 64
    assert result.artifact_bytes > 0
    assert result.license == "MIT"
    assert result.warnings == []


def test_missing_manifest_fails(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"x" * 10)
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("missing manifest.json" in e for e in result.errors)


def test_missing_placeholders_fails(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "compress.py"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    (tmp_path / "compress.py").write_text("x")
    (tmp_path / "decompress.py").write_text("x")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("placeholder" in e for e in result.errors)


def test_size_cap_enforced():
    result = precheck_artifact_dir(REFERENCE_CODEC, max_artifact_bytes=10)
    assert result.ok is False
    assert any("cap is" in e for e in result.errors)


def test_entrypoint_missing_script_warns(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "nope.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "dec.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    (tmp_path / "dec.py").write_text("x")
    result = precheck_artifact_dir(tmp_path)
    assert any("missing local file" in w for w in result.warnings)
