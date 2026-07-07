"""issue #66: artifact_ref() must fetch via a real (non-symlinked) local_dir before handing
the result to a bind-mounting runner (DockerRunner, or LocalSubprocessRunner) -- the default
huggingface_hub cache-based download returns symlinks into a separate blobs/ dir that dangle
once the snapshot directory is bind-mounted alone into a container."""

from unittest.mock import patch

from core.artifact import local_snapshot_dir
from core.state import CommitmentState
from eval.runner import LocalSubprocessRunner
from reign_worker.service import artifact_ref


def _commitment(**overrides):
    defaults = dict(
        hotkey="hk", repo="org/repo", revision="deadbeef1234", block=1,
        artifact_hash="h", artifact_bytes=10, valid=True,
    )
    return CommitmentState(**{**defaults, **overrides})


def test_artifact_ref_downloads_with_stable_local_dir_for_local_runner():
    commitment = _commitment()
    with patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = "/tmp/whatever"
        artifact_ref(commitment, LocalSubprocessRunner())
    mock_download.assert_called_once()
    kwargs = mock_download.call_args.kwargs
    assert kwargs["repo_id"] == "org/repo"
    assert kwargs["revision"] == "deadbeef1234"
    assert kwargs["local_dir"] == local_snapshot_dir("org/repo", "deadbeef1234")


def test_artifact_ref_downloads_for_any_needs_local_artifact_runner():
    # DockerRunner (and any future local-execution runner) is flagged via
    # needs_local_artifact, not an isinstance check -- a bare stand-in object with that
    # attribute must be treated identically to LocalSubprocessRunner.
    commitment = _commitment()
    stub_runner = type("StubRunner", (), {"needs_local_artifact": True})()
    with patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = "/tmp/whatever"
        artifact_ref(commitment, stub_runner)
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["local_dir"] == local_snapshot_dir("org/repo", "deadbeef1234")


def test_artifact_ref_skips_download_when_local_path_already_set():
    commitment = _commitment(local_path="/already/here")
    with patch("huggingface_hub.snapshot_download") as mock_download:
        ref = artifact_ref(commitment, LocalSubprocessRunner())
    mock_download.assert_not_called()
    assert ref.local_path == "/already/here"


def test_artifact_ref_does_not_download_for_remote_runner():
    # A runner that neither is LocalSubprocessRunner nor sets needs_local_artifact (e.g.
    # ChutesRunner) downloads inside its own remote worker -- no local fetch here at all.
    commitment = _commitment()
    remote_runner = type("RemoteRunner", (), {})()
    with patch("huggingface_hub.snapshot_download") as mock_download:
        ref = artifact_ref(commitment, remote_runner)
    mock_download.assert_not_called()
    assert ref.local_path is None
