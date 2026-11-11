import concurrent.futures
from pathlib import Path
import tempfile
import unittest
import zipfile

from posim_routing.coast_dataset import _cached_dataset, coast_dataset


class CoastDatasetTests(unittest.TestCase):
    def test_concurrent_readers_share_one_generation_and_refresh_on_change(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'land.zip'
            with zipfile.ZipFile(archive,'w') as out:
                out.writestr('land/shore.shp',b'first')
                out.writestr('land/shore.dbf',b'attributes')
            stat=archive.stat();signature=(stat.st_size,stat.st_mtime_ns)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                owners=list(pool.map(lambda _:coast_dataset(str(archive),signature),range(16)))
            self.assertTrue(all(owner is owners[0] for owner in owners))
            self.assertEqual(owners[0].dataset.read_bytes(),b'first')
            self.assertEqual(owners[0].dataset.parent.parent,archive.parent)
            with zipfile.ZipFile(archive,'w') as out:
                out.writestr('shore.shp',b'new generation')
            stat=archive.stat();updated=coast_dataset(str(archive),(stat.st_size,stat.st_mtime_ns))
            self.assertIsNot(updated,owners[0])
            self.assertEqual(updated.dataset.read_bytes(),b'new generation')
            self.assertEqual(owners[0].dataset.read_bytes(),b'first')
            _cached_dataset.cache_clear()


if __name__=='__main__':unittest.main()
