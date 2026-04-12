#!/usr/bin/env python3
"""Unit tests for native_hw + default IOSurfaceKVManager (macOS + dylib only)."""

import os
import platform
import sys
import unittest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)


def _darwin() -> bool:
    return platform.system() == "Darwin"


@unittest.skipUnless(_darwin(), "libgemma_hw is macOS-only")
class TestNativeHW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import native_hw

        cls._nh = native_hw
        if not native_hw.libgemma_hw_present():
            raise unittest.SkipTest(
                f"libgemma_hw.dylib not built — run: make -C secret-apis libgemma_hw "
                f"(expected {native_hw.libgemma_hw_dylib_path()})"
            )

    def test_capabilities_iosurface_ok(self):
        cap = self._nh.capabilities()
        self.assertTrue(cap.iosurface_create_ok, msg=self._nh.library_load_error())

    def test_sgemm_matches_numpy(self):
        import numpy as np

        a = np.random.randn(24, 24).astype(np.float32)
        b = np.random.randn(24, 24).astype(np.float32)
        c = self._nh.sgemm(a, b)
        np.testing.assert_allclose(c, a @ b, rtol=0, atol=1e-3)

    def test_metal_iosurface_selftest(self):
        self.assertTrue(self._nh.metal_iosurface_selftest())

    def test_iosurface_kv_default_manager(self):
        from hw_accel import HWAccelConfig, IOSurfaceKVManager

        self.assertTrue(HWAccelConfig().iosurface_enabled)
        mgr = IOSurfaceKVManager()
        self.assertTrue(mgr.native_zero_copy, msg=self._nh.library_load_error())
        self.assertTrue(mgr.allocate_kv_surface("t", 2048))
        buf = mgr.get_surface("t")
        self.assertIsNotNone(buf)
        buf.fill(0)
        buf[1] = 0x77
        self.assertEqual(int(mgr.get_surface("t")[1]), 0x77)
        mgr.release_all()


@unittest.skipIf(_darwin(), "macOS defaults IOSurface to on")
class TestHWAccelIOSurfaceDefaultOffNonMac(unittest.TestCase):
    def test_iosurface_disabled_by_default(self):
        from hw_accel import HWAccelConfig

        self.assertFalse(HWAccelConfig().iosurface_enabled)


if __name__ == "__main__":
    unittest.main()
