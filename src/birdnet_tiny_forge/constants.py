from pathlib import Path

import platformdirs


_CACHE_DIR = Path(platformdirs.user_cache_dir("tinyforge", ensure_exists=True))