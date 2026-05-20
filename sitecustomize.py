"""Process-wide startup settings for hosted Streamlit deployments."""

from __future__ import annotations

import os


# Some transitive dependencies still load older generated protobuf modules in
# hosted environments. This must be set before protobuf is imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
