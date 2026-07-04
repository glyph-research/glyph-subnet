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

Round-trip compatibility still needs a live Docker/GPU validation pass with the reference codec
and a representative neural codec before making a stricter custom seccomp profile mandatory.
