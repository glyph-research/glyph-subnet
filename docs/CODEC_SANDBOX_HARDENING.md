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
- Docker's default seccomp profile unless `--docker-seccomp-profile PATH` is provided.

The scratch directory is made writable before launch because the sandbox UID is intentionally
not the validator host user. The artifact tree is made readable/traversable for that UID and
then mounted read-only. The scratch directory is temporary and only contains that phase's input
and output files.

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

Note this shifts part of the security model: static source review (below) has much less to
say about a codec that ships an opaque image -- the actual compress/decompress code and deps
are baked into layers, not present as reviewable files in the mounted artifact dir. Isolation
here is enforced by the runtime warmup/seal/benchmark lifecycle, not by capping what a miner
may install.

## Split-Worker Isolation (Default Path)

The local subprocess and Chutes helper retain the existing `unshare --net` network isolation.
They can also apply `setpriv` when the operator sets both `GLYPH_CODEC_PRIVDROP_UID` and
`GLYPH_CODEC_PRIVDROP_GID`. Set `GLYPH_CODEC_REQUIRE_PRIVDROP=1` to fail closed if `setpriv`
is unavailable. This is intentionally environment-gated because Chutes and some CI containers
may not permit UID/GID changes even when network isolation is available.

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
  and fails otherwise); the scored entrypoint's own attempted network connect fails (proving
  the seal actually happened, not just a flag); a warmup that never signals ready is killed
  and raises rather than hanging; a digest-pinned image round-trips end to end. Live-verified
  against a real pushed-and-digest-pinned image (`samples/docker_codec_template/`) through a
  throwaway local registry.

Round-trip compatibility still needs a live Docker/GPU validation pass with the reference codec
and a representative neural codec before making a stricter custom seccomp profile mandatory.
The miner-published-image lifecycle above is verified on CPU-only hardware (this sandbox has
no GPU); a live GPU pass against a real torch-based image is still open, same caveat as the
rest of this doc.
