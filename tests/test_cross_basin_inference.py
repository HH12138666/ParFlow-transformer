"""Regression checks for target-grid geometry and overlap reconstruction."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from parflow.tools.io import read_pfb, write_pfb

from model_deployment.common import TimingStats, validate_config
from model_deployment.engine import pfb_grid, predict_full_block, stitch_predictions, write_prediction_block


class CrossBasinInferenceTests(unittest.TestCase):
    def spec(self, window):
        return SimpleNamespace(model_config=dict(input_channels=45, height=146, width=252,
            space_h=window[0], space_w=window[1], out_channels=1))

    def test_tiled_target_accepts_different_domain(self):
        validate_config(self.spec((64, 80)), SimpleNamespace(c_in=45, height=202, width=118))

    def test_full_domain_rejects_different_shape(self):
        with self.assertRaisesRegex(ValueError, 'Spatial size mismatch'):
            validate_config(self.spec((None, None)), SimpleNamespace(c_in=45, height=202, width=118))

    def test_oversized_window_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exceeds data size'):
            validate_config(self.spec((64, 128)), SimpleNamespace(c_in=45, height=202, width=118))

    def test_uncovered_cells_rejected(self):
        with self.assertRaisesRegex(ValueError, 'uncovered cells'):
            predict_full_block(None, None, [], self.spec((2, 2)), SimpleNamespace(height=3, width=3),
                               SimpleNamespace(patch_batch_size=1), TimingStats(), 1)

    def test_overlap_average(self):
        patches = np.stack((np.ones((2, 1, 3, 3)), np.full((2, 1, 3, 3), 3.)))
        total, count = np.zeros((2, 1, 3, 5)), np.zeros((2, 1, 3, 5), dtype=int)
        stitch_predictions(patches, total, count, [(0, 0), (0, 2)])
        np.testing.assert_array_equal((total / count)[0, 0], np.tile((1., 1., 2., 3., 3.), (3, 1)))

    def test_written_pfb_retains_grid(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            source = root / 'source.pfb'
            grid = dict(x=10., y=20., z=0., dx=961.72, dy=961.72, dz=200.)
            values = np.arange(24, dtype=float).reshape(2, 3, 4)
            write_pfb(str(source), values, dist=False, **grid)
            sources = SimpleNamespace(files=[str(source)], hours=[20210012],
                                      rel_paths=[Path('2021/press_20210012.pfb')])
            write_prediction_block(values[None], 0, 0, 1, root / 'output', sources, 2, TimingStats())
            output = root / 'output/2021/pred_press_20210012.pfb'
            self.assertEqual(pfb_grid(output), grid)
            np.testing.assert_array_equal(read_pfb(str(output)), values)


if __name__ == '__main__':
    unittest.main()
