"""Source-checkout entry point; accepts the same subcommands as inv-nasn-ct."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inv_framework.applications.nasn_ct.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
