"""Entrypoint to launch the TRADE9 Claymorphic Web Studio and Terminal."""

import sys
from pathlib import Path

# Add src to sys.path if not installed
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from trady.studio.server import start_studio_server  # noqa: E402

if __name__ == "__main__":
    port = 8090
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_studio_server(port=port)
