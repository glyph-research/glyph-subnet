import json
from pathlib import Path
from unittest.mock import patch

from core.artifact import local_snapshot_dir
from validation.precheck import precheck_artifact_dir, precheck_codec

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


def _manifest_with_image(image):
    return {
        "schema_version": 1,
        "entrypoints": {
            "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
            "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
        },
        "image": image,
    }


def test_manifest_image_mutable_tag_fails_precheck(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest_with_image("ghcr.io/user/mycodec:latest")))
    (tmp_path / "compress.py").write_text("x")
    (tmp_path / "decompress.py").write_text("x")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("digest" in e for e in result.errors)


def test_manifest_image_digest_pinned_passes_precheck(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(_manifest_with_image("ghcr.io/user/mycodec@sha256:" + "a" * 64))
    )
    (tmp_path / "compress.py").write_text("x")
    (tmp_path / "decompress.py").write_text("x")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is True


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


def _write_basic_manifest(directory: Path) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "compress.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )


def test_network_import_fails_security_precheck(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        "import argparse\n"
        "import requests\n"
        "p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output')\n"
        "a=p.parse_args();open(a.output,'wb').write(open(a.input,'rb').read())\n"
    )
    (tmp_path / "decompress.py").write_text("open(__import__('sys').argv[-1], 'wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("network/cloud import 'requests'" in e for e in result.errors)


def test_dynamic_network_import_fails_security_precheck(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        "import importlib\n"
        "importlib.import_module('urllib.request')\n"
        "open('out','wb').write(b'x')\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("dynamic import of network module 'urllib.request'" in e for e in result.errors)


def test_shell_network_command_fails_security_precheck(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["bash", "-c", "curl https://example.test/{input} > {output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("inline code execution" in e for e in result.errors)
    assert any("external URL/protocol" in e for e in result.errors)


def test_subprocess_shell_true_fails_security_precheck(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        "import subprocess\n"
        "subprocess.run('echo x', shell=True)\n"
        "open('out','wb').write(b'x')\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("subprocess shell=True" in e for e in result.errors)


def test_subprocess_network_command_fails_security_precheck(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        "import subprocess\n"
        "subprocess.run(['curl', 'https://example.test'])\n"
        "open('out','wb').write(b'x')\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("subprocess network command 'curl'" in e for e in result.errors)


def test_python_comment_and_docstring_urls_pass_security_precheck(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        '"""Model notes: https://example.test/paper."""\n'
        "# reference: https://arxiv.org/abs/1234\n"
        "def helper():\n"
        '    """More notes: https://github.com/org/repo."""\n'
        "    return b'x'\n"
        "open('out','wb').write(helper())\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is True


def test_python_executable_url_literal_fails_security_precheck(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        "endpoint = 'https://example.test/upload'\n"
        "open('out','wb').write(endpoint.encode())\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("external URL/protocol literal" in e for e in result.errors)


def test_shell_comments_do_not_trip_network_scanner(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["bash", "compress.sh", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    (tmp_path / "compress.sh").write_text(
        "# reference: https://example.test/model\n"
        "# curl https://example.test/upload\n"
        "printf '%s' safe > \"$4\"\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is True


def test_shell_executable_url_still_fails_security_precheck(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["bash", "compress.sh", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "decompress.py", "--input", "{input}", "--output", "{output}"],
                },
            }
        )
    )
    (tmp_path / "compress.sh").write_text("curl 'https://example.test/upload' > \"$4\"\n")
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is False
    assert any("external URL/protocol" in e for e in result.errors)
    assert any("network command 'curl'" in e for e in result.errors)


def test_python_identifiers_do_not_trip_command_scanner(tmp_path):
    _write_basic_manifest(tmp_path)
    (tmp_path / "compress.py").write_text(
        "host = 'local'\n"
        "nc = 3\n"
        "open('out','wb').write(f'{host}{nc}'.encode())\n"
    )
    (tmp_path / "decompress.py").write_text("open('out','wb').write(b'x')\n")
    result = precheck_artifact_dir(tmp_path)
    assert result.ok is True


def test_precheck_codec_downloads_with_stable_local_dir(tmp_path):
    # issue #66: must not use the default cache-based (symlink-into-blobs/) download --
    # local_dir materializes real files, and reusing the same stable path artifact_ref() uses
    # means a later real fetch for execution can skip re-downloading what precheck already
    # verified.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoints": {
                    "compress": ["python3", "c.py", "--input", "{input}", "--output", "{output}"],
                    "decompress": ["python3", "d.py", "--input", "{input}", "--output", "{output}"],
                },
                "license": "MIT",
            }
        )
    )
    (tmp_path / "c.py").write_text("x")
    (tmp_path / "d.py").write_text("x")

    with patch("huggingface_hub.HfApi") as mock_api, patch("huggingface_hub.snapshot_download") as mock_download:
        mock_api.return_value.repo_info.return_value = None
        mock_download.return_value = str(tmp_path)
        precheck_codec("org/repo", "deadbeef1234", max_artifact_bytes=1000)

    mock_download.assert_called_once_with(
        repo_id="org/repo", revision="deadbeef1234", local_dir=local_snapshot_dir("org/repo", "deadbeef1234")
    )
