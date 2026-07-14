"""Glyph lossless neural text-compression rolling-winner subnet."""


def _version_key_from_version(version: str) -> int:
    """major.minor.patch -> major * 1000 + minor * 10 + patch."""
    major, minor, patch = (int(part) for part in version.split("."))
    return major * 1000 + minor * 10 + patch


__version__ = "1.1.0"
__version_key__ = _version_key_from_version(__version__)
