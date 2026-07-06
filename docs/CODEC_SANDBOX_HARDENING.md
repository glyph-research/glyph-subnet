# Codec Sandbox Hardening

Issue #31 tracks the residual container-escape risk after the existing split-worker,
network-isolated, secret-scrubbed codec execution boundary.

## Chosen Mechanism

The default validator path is `DockerRunner`. Each compress and decompress phase runs in a
fresh `docker run --rm` container with:

- `--network none` while corpus bytes or compressed blobs are present,
- a read-only artifact mount and an ephemeral scratch mount,
- `--user 65534:65534` so untrusted codec code does not run as root,
- `--cap-drop ALL` so Linux capabilities are unavailable inside the codec process,
- `--security-opt no-new-privileges:true`,
- Docker's default seccomp profile unless `--docker-seccomp-profile PATH` is provided,
- `--ulimit fsize=<core.constants.SCRATCH_CAP_BYTES>` plus a background disk watchdog (below).

The scratch directory is made writable before launch because the sandbox UID is intentionally
not the validator host user. The artifact tree is made readable/traversable for that UID and
then mounted read-only. The scratch directory is temporary and only contains that phase's input
and output files.

## Scratch Disk Cap (issue #54)

The scratch mount is an unbounded host bind mount by construction (needed so the runner can
read compressed/decompressed output back off the host side), so nothing stopped a codec from
filling the validator's disk until this was added. Two complementary mechanisms, both driven
by `core.constants.SCRATCH_CAP_BYTES` (117 GiB default -- a host-safety limit, not a
consensus-relevant value like `DOCKER_REFERENCE_GPU`/`REFERENCE_SKU`, so `DockerRunner`
exposes it as an overridable `scratch_cap_bytes` constructor parameter mainly so tests don't
need to write real gigabytes):

- **`--ulimit fsize=<cap>`** on the container: a kernel-enforced per-file size limit
  (`RLIMIT_FSIZE`). Catches a single oversized file instantly (`EFBIG`), with zero polling
  overhead, but doesn't bound the sum of many smaller files.
- **`_DiskWatchdog`** (a background thread, `eval/runner_docker.py`): polls the scratch
  directory's total size every 2s while a blocking `docker run`/`docker exec` call is in
  progress and kills the container the instant the cumulative total exceeds the cap -- the
  many-small-files case the per-file ulimit alone wouldn't catch. `subprocess.run(...,
  timeout=...)` alone only bounds wall-clock time, not how much disk a codec could fill
  before that timeout elapses (the compress/decompress budget can be tens of minutes).

Applies to **both** DockerRunner paths: the classic single-shot `_run_container` (one
watchdog per phase) and the miner-published-image `_run_networked_lifecycle` (one watchdog
spanning both the networked warmup -- where `HOME`/`TMPDIR`/`XDG_CACHE_HOME` all point at
`/scratch`, so pip installs / weight downloads count against the same budget -- and the
sealed benchmark, since both phases share the same scratch mount).

## Miner-Published Images: Networked Warmup (issue #48)

A manifest may declare its own digest-pinned `image` instead of running inside the
operator-supplied default. That image is allowed network access, but only during a bounded
**warmup** phase, before the container has ever seen any eval data:

1. **Warmup (network ON):** the container starts detached (`docker run -d`), on its own
   per-invocation bridge network (not the shared default bridge), with an EMPTY scratch mount
   -- there is nothing present yet for a malicious codec to exfiltrate even with full network
   access. It installs deps / downloads or loads weights, then signals readiness (either a
   bounded `warmup.command` exiting 0, or the image's own long-running process creating
   `warmup.ready_file`), bounded by `warmup.timeout_secs`. Exceeding the deadline (or a
   nonzero exit) kills the container and fails the codec closed -- it never runs unbounded.
2. **Seal:** `docker network disconnect` is called on the container the instant it's ready.
   From this point on it has no network route at all, for either compress or decompress.
3. **Benchmark:** only now is the eval input written into the (already-mounted, so-far-empty)
   scratch directory, and the scored entrypoint is `docker exec`'d into the same sealed
   container -- offline, same GPU/RAM caps and wall-clock budget as the default path.

The ordering is the whole security argument: the container's network is severed **before**
any eval byte is written anywhere the container can reach, not merely disabled by a flag it
could race. Compress and decompress each get their own fresh container and warmup (same
stash-defeat invariant as the default path -- see below) -- a manifest with no `image` is
entirely unaffected and keeps the original single-shot `--network none`-from-start path.

**A subtlety verified during review (PR #53):** a socket a codec opened *during* warmup and
kept open across the seal keeps reporting `send()` success afterward -- TCP buffers writes
into the local kernel send buffer regardless of whether the underlying interface still
exists, so "does `send()` raise" is not the right thing to check, and `docker network
disconnect --force` does not change this either. What was verified directly (both by hand
against a controlled receiver, and in `test_established_connection_stops_delivering_after_seal`
against a real external endpoint) is the property that actually matters: no round trip -- no
bytes genuinely delivered anywhere -- completes once the interface is severed, even on a
connection that was alive and working moments before. The seal is real at the delivery layer;
it just doesn't surface as an immediate error at the socket API layer.

Note this shifts part of the security model: static source review (below) has much less to
say about a codec that ships an opaque image -- the actual compress/decompress code and deps
are baked into layers, not present as reviewable files in the mounted artifact dir. Isolation
here is enforced by the runtime warmup/seal/benchmark lifecycle, not by capping what a miner
may install.

## Split-Worker Isolation (Default Path)

The local subprocess and Chutes helper retain the existing `unshare --net` network isolation.
`LocalSubprocessRunner` (`eval/runner.py`) can also apply `setpriv` when the operator sets
both `GLYPH_CODEC_PRIVDROP_UID` and `GLYPH_CODEC_PRIVDROP_GID`, and fail closed via
`GLYPH_CODEC_REQUIRE_PRIVDROP=1` -- intentionally opt-in there, since some CI containers may
not permit UID/GID changes even when network isolation is available.

**The Chutes eval runner (`eval/glyph_eval_runner.py`) drops privileges by default now**
(issue #57): it previously only applied `setpriv` when those same env vars were explicitly
set, and they were never set on the deployed Chutes path -- the untrusted codec ran there as
**root**, isolated only by `unshare --net`, materially weaker than the Docker path's `--user
65534:65534`. It now requests `--reuid 65534 --regid 65534` (matching Docker's non-root
default) unconditionally, still overridable via `GLYPH_CODEC_PRIVDROP_UID`/`_GID`, and fails
closed (`RunnerError`) if `setpriv` is unavailable -- set `GLYPH_CODEC_ALLOW_UNPRIVILEGED=1`
to opt back into the old soft-fail behavior for an environment that genuinely can't apply it.
Since `tempfile.mkdtemp()` defaults to mode 0700, `_run_codec` now also makes the artifact
directory tree traversable/readable for that dropped uid before exec'ing into it (mirroring
`DockerRunner`'s `_allow_sandbox_read_tree`) -- without this the codec couldn't even open its
own entrypoint script once root was actually dropped, caught while adding this fix's own
tests.

**Also fixed (issue #57): zip-slip in the artifact download.** `_hf_snapshot`'s stdlib-only
`snapshot_download` equivalent joined a miner-controlled HF tree-listing path directly onto
the download destination with no containment check; a `rel` containing `../` (or an absolute
path) could write outside the artifact directory. `_safe_join` now resolves and rejects any
entry that would escape the destination before ever writing to disk.

## Seccomp Rationale

The current baseline uses Docker's default seccomp allowlist rather than a custom minimal
allowlist. That profile keeps normal Python, compression libraries, CUDA user-space calls,
`mmap`, threads, futexes, and file I/O available, while denying high-risk syscall families
such as kernel module loading, raw kernel keyring operations, broad namespace changes, and
other privileged kernel-control surfaces.

A stricter profile should be reviewed against representative neural codecs before it becomes
the default. In particular, do not remove syscall families used by:

- Python startup and dynamic library loading,
- `mmap`, `mprotect`, shared memory, and allocator behavior,
- thread creation and synchronization,
- CUDA/NVIDIA user-space driver interaction,
- ordinary read/write/stat/open/rename operations inside `/scratch`.

When a stricter profile is approved, pass it with:

```bash
glyph-validator --docker-seccomp-profile /path/to/seccomp-codec.json ...
```

## Verification

Unit coverage asserts:

- Docker codec commands include non-root user, cap drop, no-new-privileges, network isolation,
  and optional seccomp profile pass-through.
- Subprocess hardening requires UID/GID to be configured together.
- `unshare --net` remains the outer wrapper, with `setpriv` applied inside the isolated network
  namespace.
- Miner-published-image lifecycle (`tests/test_runner_docker_warmup.py`, real Docker, no GPU
  needed): a mutable-tag image is rejected before any container runs; the eval input is
  genuinely absent from the scratch mount during warmup (a warmup command that asserts this
  and fails otherwise); the scored entrypoint's own attempted network connect fails, with a
  non-vacuous warmup-side check that real egress existed in the first place (proving the seal
  actually happened, not just a flag, and not a false pass from a host with no internet); a
  connection opened during warmup and held open across the seal genuinely stops delivering
  round trips afterward (ground-truth delivery check against a real external endpoint, not a
  `send()`-return-value check -- see the subtlety noted above); a warmup that never signals
  ready is killed and raises rather than hanging; a digest-pinned image round-trips end to
  end. The image-digest regex itself is unit-tested (`tests/test_artifact.py`) against
  mutable tags, trailing/leading whitespace, a trailing newline, and an embedded second
  `@sha256:...` being swallowed into the name portion. Live-verified against a real
  pushed-and-digest-pinned image (`samples/docker_codec_template/`) through a throwaway local
  registry.
- Scratch disk cap (`tests/test_runner_docker.py`, `tests/test_runner_docker_warmup.py`, real
  Docker): a codec writing many small files that cumulatively exceed a (tiny, test-only)
  `scratch_cap_bytes` is killed and fails with a clear "disk cap" error on both the classic
  single-shot path and the networked lifecycle -- during the sealed benchmark AND during the
  networked warmup itself (a runaway pip install / weights download); a single file exceeding
  the per-file `--ulimit fsize` cap fails instantly via the kernel (`EFBIG`), the complementary
  case the watchdog alone wouldn't catch as fast; a normal codec comfortably under the cap
  still round-trips bit-exact on both paths.
- Chutes eval runner hardening (`tests/test_glyph_eval_runner.py`, issue #57): privilege drop
  is requested by default with no env vars set (uid/gid default to 65534, overridable);
  fails closed when `setpriv` is unavailable, with an explicit opt-out
  (`GLYPH_CODEC_ALLOW_UNPRIVILEGED`) for environments that genuinely can't apply it. A real
  (not mocked) end-to-end run confirms the codec observes a non-root uid AND can still read
  its own entrypoint despite `tempfile.mkdtemp()`'s default 0700 mode. `_safe_join` rejects a
  crafted artifact tree-listing path (`../`, absolute) before ever writing to disk, covering
  the zip-slip fix.

Round-trip compatibility still needs a live Docker/GPU validation pass with the reference codec
and a representative neural codec before making a stricter custom seccomp profile mandatory.
The miner-published-image lifecycle above is verified on CPU-only hardware (this sandbox has
no GPU); a live GPU pass against a real torch-based image is still open, same caveat as the
rest of this doc.
