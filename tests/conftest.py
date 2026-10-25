"""pytest 公共配置：把项目根目录放进 sys.path，让 tests/ 能用 `import api.xxx`。

直接 `pytest tests/` 时 sys.path[0] 是 tests/，找不到 api/tools/utils 包，这里统一补齐。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
