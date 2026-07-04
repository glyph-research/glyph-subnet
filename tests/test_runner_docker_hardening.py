import subprocess

from eval.runner import ResourceCaps
from eval.runner_docker import DockerRunner


def test_docker_runner_applies_default_hardening_flags(monkeypatch, tmp_path):
    seen = {}

    def _run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("eval.runner_docker.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run)

    runner = DockerRunner(gpu=False)
    runner._run_container(["python3", "compress.py"], tmp_path, tmp_path, ResourceCaps(network=False))

    cmd = seen["cmd"]
    assert "--user" in cmd
    assert cmd[cmd.index("--user") + 1] == "65534:65534"
    assert "--cap-drop" in cmd
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in cmd
    assert "no-new-privileges:true" in cmd
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"


def test_docker_runner_passes_seccomp_profile(monkeypatch, tmp_path):
    seen = {}

    def _run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("eval.runner_docker.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("eval.runner_docker.subprocess.run", _run)

    runner = DockerRunner(gpu=False, seccomp_profile="/etc/glyph/seccomp-codec.json")
    runner._run_container(["python3", "compress.py"], tmp_path, tmp_path, ResourceCaps(network=True))

    assert "seccomp=/etc/glyph/seccomp-codec.json" in seen["cmd"]
