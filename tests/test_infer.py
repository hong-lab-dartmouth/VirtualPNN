"""Tests for infer.py. They need no weights, GPU or data.
Coverage: tile coverage, seam blending, the intensity round-trip and fold discovery.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import infer  # noqa: E402


class TestTiling:
    def test_origins_cover_the_whole_extent(self):
        for extent in (256, 300, 512, 700, 1536, 2049):
            for tile, stride in ((256, 256), (256, 224), (256, 128)):
                xs = infer.tile_origins(extent, tile, stride)
                covered = np.zeros(extent, dtype=bool)
                for x in xs:
                    covered[x:x + tile] = True
                assert covered.all(), f"gap at extent={extent} tile={tile} stride={stride}"

    def test_last_tile_is_flush_with_the_edge(self):
        # otherwise the final strip is padding, and the model sees reflected pixels
        for extent in (300, 700, 1000):
            xs = infer.tile_origins(extent, 256, 256)
            assert xs[-1] == extent - 256

    def test_smaller_than_one_tile(self):
        assert infer.tile_origins(200, 256, 256) == [0]

    def test_no_origin_runs_past_the_edge(self):
        for extent in (257, 511, 999):
            for x in infer.tile_origins(extent, 256, 192):
                assert 0 <= x <= extent - 256


class TestBlending:
    def test_zero_overlap_is_a_flat_window(self):
        w = infer.blend_window(256, 0)
        assert w.shape == (256, 256) and np.all(w == 1.0)

    def test_taper_is_positive_and_symmetric(self):
        w = infer.blend_window(256, 32)
        assert (w > 0).all(), "a zero weight would divide by ~0 when merging"
        assert np.allclose(w, w[::-1, :]) and np.allclose(w, w[:, ::-1])

    def test_interior_is_unweighted(self):
        w = infer.blend_window(256, 32)
        assert np.allclose(w[64:192, 64:192], 1.0)


class TestIntensityRoundTrip:
    def test_tensor_range(self):
        img = Image.fromarray(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))
        t = infer.to_tensor(img)
        assert t.shape == (3, 1, 2)
        assert pytest.approx(float(t.min())) == -1.0
        assert pytest.approx(float(t.max())) == 1.0

    def test_round_trip_is_lossless_for_uint8(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        back = infer.to_image(infer.to_tensor(Image.fromarray(a)))
        assert np.array_equal(a, back)

    def test_out_of_range_is_clamped_not_wrapped(self):
        t = torch.tensor([[[-4.0]], [[4.0]], [[0.0]]])
        out = infer.to_image(t)
        assert out[0, 0, 0] == 0 and out[0, 0, 1] == 255


class TestFoldDiscovery:
    def _make(self, tmp_path, names):
        for n in names:
            d = tmp_path / n
            d.mkdir(parents=True)
            (d / "latest_net_G.pth").write_bytes(b"")
        return tmp_path

    def test_fold_number_comes_from_the_fold_marker(self, tmp_path):
        # The fold number comes from the _Fold marker, so VirtualPNN_AV6_Fold3 is fold 3.
        d = self._make(tmp_path, [f"VirtualPNN_AV6_Fold{k}" for k in (1, 2, 3)])
        assert sorted(infer.find_folds(d, None)) == [1, 2, 3]

    def test_explicit_selection(self, tmp_path):
        d = self._make(tmp_path, [f"VirtualPNN_AV6_Fold{k}" for k in (1, 2, 3)])
        assert sorted(infer.find_folds(d, [1, 3])) == [1, 3]

    def test_missing_fold_exits_with_a_message(self, tmp_path):
        d = self._make(tmp_path, ["VirtualPNN_AV6_Fold1"])
        with pytest.raises(SystemExit):
            infer.find_folds(d, [9])

    def test_empty_weights_dir_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            infer.find_folds(tmp_path, None)

    def _fetched_layout(self, tmp_path):
        """Empty files named as fetch_weights.sh lays out the weights archive."""
        runs = ("VirtualPNN_AV6", "VirtualPNN_noAV", "CUT", "AIFFPE")
        d = self._make(tmp_path, [f"{r}_Fold{k}" for r in runs for k in range(1, 6)])
        files = [f"CycleGAN_Fold{k}/latest_net_G_{x}.pth" for k in range(1, 6) for x in "AB"]
        files += [f"aiffpe/5_net_{x}.pth" for x in "GDF"]
        files += [f"aiffpe_weights/5_net_{x}.pth" for x in ("G", "D", "F", "G_A", "G_B", "D_A", "D_B")]
        for rel in files:
            (d / rel).parent.mkdir(exist_ok=True)
            (d / rel).write_bytes(b"")
        return d

    def test_fetched_layout_selects_the_main_model_by_default(self, tmp_path):
        # Keyed by fold number alone, VirtualPNN_noAV_Fold<k> sorts after VirtualPNN_AV6_Fold<k>
        # and replaced it: the default command ran VirtualPNN-P instead of the paper's model.
        d = self._fetched_layout(tmp_path)
        folds = infer.find_folds(d, None)
        assert {k: p.parent.name for k, p in folds.items()} == \
            {k: f"VirtualPNN_AV6_Fold{k}" for k in range(1, 6)}

    def test_fetched_layout_other_models_by_run_name(self, tmp_path):
        d = self._fetched_layout(tmp_path)
        for run in ("VirtualPNN_noAV", "CUT", "AIFFPE"):
            folds = infer.find_folds(d, [2, 4], run)
            assert {k: p.parent.name for k, p in folds.items()} == {2: f"{run}_Fold2", 4: f"{run}_Fold4"}

    def test_two_folders_for_one_fold_is_an_error(self, tmp_path):
        d = self._fetched_layout(tmp_path)
        with pytest.raises(SystemExit):
            infer.find_folds(d, None, "")


class TestInputCollection:
    def test_finds_images_recursively_and_ignores_other_files(self, tmp_path):
        (tmp_path / "sub").mkdir()
        for rel in ("a.png", "sub/b.tif", "sub/c.JPG"):
            Image.new("RGB", (4, 4)).save(tmp_path / rel)
        (tmp_path / "notes.txt").write_text("ignore me")
        assert len(infer.collect_inputs(tmp_path)) == 3

    def test_single_file(self, tmp_path):
        f = tmp_path / "one.png"
        Image.new("RGB", (4, 4)).save(f)
        assert infer.collect_inputs(f) == [f]

    def test_directory_without_images_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            infer.collect_inputs(tmp_path)


class TestGenerator:
    def test_architecture_matches_the_released_checkpoints(self):
        """The released generators' key set matches the architecture infer.py builds."""
        from models import networks
        net = networks.define_G(infer.ARCH["input_nc"], infer.ARCH["output_nc"],
                                infer.ARCH["ngf"], infer.ARCH["netG"], infer.ARCH["norm"],
                                infer.ARCH["use_dropout"], infer.ARCH["init_type"],
                                infer.ARCH["init_gain"], infer.ARCH["no_antialias"],
                                infer.ARCH["no_antialias_up"], gpu_ids=[])
        keys = set(net.state_dict())
        # the AI-FFPE spatial attention block is part of every published checkpoint
        assert {"SAB.conv1.weight", "SAB.conv2.weight", "SAB.conv3.weight"} <= keys
        assert len(keys) == 58, f"generator key count changed: {len(keys)} != 58"

    def test_translate_preserves_input_size(self):
        """A stub generator, so this tests the tiling and merge, not the network."""
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x
        for size in ((300, 200), (256, 256), (513, 97)):
            img = Image.new("RGB", size, (200, 150, 190))
            out = infer.translate([Identity()], img, 256, 0, torch.device("cpu"), 4)
            assert out.size == size

    def test_translate_is_faithful_through_an_identity_network(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (300, 220, 3), dtype=np.uint8)

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x
        out = infer.translate([Identity()], Image.fromarray(a), 256, 32,
                              torch.device("cpu"), 4)
        # blending is a weighted mean of identical tiles, so it must reproduce the input
        assert np.abs(np.asarray(out).astype(int) - a.astype(int)).max() <= 1
