"""Seekable coastline data shared by readers of one archive generation."""
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import threading
import zipfile


class CoastDataset:
    def __init__(self, archive):
        archive = Path(archive)
        # Keep the expanded source beside its archive, including on external
        # storage. GDAL random access inside a compressed SHP repeatedly
        # inflates preceding records and can hold the Python interpreter lock.
        self._cache = tempfile.TemporaryDirectory(prefix='.posim-coast-', dir=archive.parent)
        try:
            with zipfile.ZipFile(archive) as source:
                member = next(name for name in source.namelist() if name.endswith('.shp'))
                self.member = member
                stem = member[:-4]
                for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix'):
                    name = stem + suffix
                    if name not in source.namelist():
                        continue
                    destination = Path(self._cache.name) / Path(name).name
                    with source.open(name) as incoming, destination.open('wb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                self.dataset = Path(self._cache.name) / Path(member).name
        except BaseException:
            self._cache.cleanup()
            raise


_cache_lock = threading.Lock()


@lru_cache(maxsize=2)
def _cached_dataset(archive, signature):
    return CoastDataset(archive)


def coast_dataset(archive, signature):
    """Retain one expanded source per generation, including concurrent misses."""
    with _cache_lock:
        return _cached_dataset(archive, signature)
