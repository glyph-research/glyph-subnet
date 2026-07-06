"""Runs once per container, with network access, before the network is severed.

Download/load weights, warm up any lazy imports, etc. here -- must exit 0 to signal ready.
A nonzero exit or exceeding manifest.json's warmup.timeout_secs fails the round closed.
"""

if __name__ == "__main__":
    # e.g.: load your model into VRAM here so compress.py/decompress.py start warm.
    print("warmup complete")
