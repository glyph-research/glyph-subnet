"""The untrusted-codec subprocess environment: offline-pinned and secret-scrubbed.

These guard the exfiltration boundary -- a malicious codec/dep/weights can't read host secrets
and can't reach the network (offline HF + NO_PROXY complement the `unshare --net` isolation).
"""

from eval.runner import _SUBPROCESS_ENV_ALLOWLIST, _subprocess_env

_INJECTED = {
    "HOME", "XDG_CACHE_HOME", "TMPDIR", "NO_PROXY",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
}


def test_subprocess_env_pins_hf_offline_and_blocks_proxy(tmp_path):
    env = _subprocess_env(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["NO_PROXY"] == "*"


def test_subprocess_env_scrubs_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUTES_API_KEY", "cpk_secret_value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("PATH", "/usr/bin")  # allowlisted -> kept
    env = _subprocess_env(tmp_path)
    # No secret leaks into the codec's environment.
    assert "CHUTES_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "HF_TOKEN" not in env
    assert env["PATH"]  # allowlisted survives
    # Nothing in the env is outside the allowlist or the injected isolation vars.
    for key in env:
        assert key in _SUBPROCESS_ENV_ALLOWLIST or key in _INJECTED, f"unexpected env key {key}"
