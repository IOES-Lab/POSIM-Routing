"""A source outage can reduce coverage, never invent missing depths."""
import unittest
from posim_routing.regional import RegionalNavigator


class RegionalOutage(unittest.IsolatedAsyncioTestCase):
    async def test_offline_baseline_retries_a_smaller_verified_region(self):
        widths=[]
        async def rpc(path,body=None,timeout=10):
            if body:
                widths.append(body['width_m'])
                return dict(id=str(body['width_m']),status='failed' if len(widths)==1 else 'ready')
            return dict(baseline_status=dict(status='unavailable'))
        job=await RegionalNavigator(rpc).terrain_region([118.,24.],20000)
        self.assertEqual(widths,[20000,6000])
        self.assertEqual(job['status'],'ready')

    async def test_other_generation_failures_do_not_trigger_a_retry(self):
        widths=[]
        async def rpc(path,body=None,timeout=10):
            if body:
                widths.append(body['width_m']);return dict(id='failed',status='failed')
            return dict(baseline_status=dict(status='available'))
        with self.assertRaisesRegex(RuntimeError,'regional_terrain_unavailable'):
            await RegionalNavigator(rpc).terrain_region([118.,24.],20000)
        self.assertEqual(widths,[20000])

    async def test_smaller_region_with_missing_data_still_holds(self):
        widths=[]
        async def rpc(path,body=None,timeout=10):
            if body:
                widths.append(body['width_m']);return dict(id='failed',status='failed')
            return dict(baseline_status=dict(status='unavailable'))
        with self.assertRaisesRegex(RuntimeError,'regional_terrain_unavailable'):
            await RegionalNavigator(rpc).terrain_region([118.,24.],20000)
        self.assertEqual(widths,[20000,6000])


if __name__=='__main__':unittest.main()
