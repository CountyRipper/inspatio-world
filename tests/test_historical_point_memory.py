import os
import tempfile
import unittest

import torch

from utils.historical_point_memory import (
    DenseGeneratedPointMemory,
    RGBPointMemory,
    calibrate_depth_scale,
    compute_depth_confidence,
    dense_point_count,
    fuse_reference_and_history,
    latent_block_to_pixel_span,
    latent_keyframe_indices,
)


class FakeBlockDecoder:
    device = torch.device("cpu")

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.calls = 0

    def decode(self, latent):
        frames = 9 if self.calls == 0 else 12
        self.calls += 1
        return torch.full((1, frames, 3, self.height, self.width), 0.75)


class FakeDepthEstimator:
    device = torch.device("cpu")

    def __init__(self):
        self.last_native_shape = None
        self.last_processed_shape = None
        self.last_intrinsics_shape = None
        self.last_extrinsics_shape = None
        self.last_peak_memory_gb = 0.0
        self.block_calls = 0

    def estimate(self, rgb, output_size, output_device):
        self.last_native_shape = output_size
        return torch.ones(output_size, device=output_device)

    def estimate_block(self, rgb, output_size, output_device):
        self.block_calls += 1
        frames = rgb.shape[0]
        height, width = output_size
        self.last_native_shape = (frames, height, width)
        self.last_processed_shape = (frames, height, width, 3)
        self.last_intrinsics_shape = (frames, 3, 3)
        self.last_extrinsics_shape = (frames, 3, 4)
        depths = torch.ones((frames, height, width), device=output_device)
        K = torch.tensor([
            [30.0, 0.0, (width - 1) / 2.0],
            [0.0, 30.0, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ], device=output_device).repeat(frames, 1, 1)
        extrinsics = torch.zeros((frames, 3, 4), device=output_device)
        return rgb.to(output_device), depths, K, extrinsics


class HistoricalPointMemoryTest(unittest.TestCase):
    def setUp(self):
        self.height = 24
        self.width = 32
        self.K = torch.tensor([
            [30.0, 0.0, (self.width - 1) / 2.0],
            [0.0, 30.0, (self.height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ])
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, self.height),
            torch.linspace(0, 1, self.width),
            indexing="ij",
        )
        self.rgb = torch.stack([xx, yy, torch.full_like(xx, 0.5)], dim=0)
        self.depth = torch.ones((self.height, self.width))
        self.pose = torch.eye(4)
        self.mask = torch.ones((self.height, self.width), dtype=torch.bool)

    def make_memory(self, **kwargs):
        return RGBPointMemory(
            height=self.height,
            width=self.width,
            device=torch.device("cpu"),
            K=self.K,
            voxel_size=kwargs.pop("voxel_size", 0.0),
            max_points=kwargs.pop("max_points", self.height * self.width),
            point_size=kwargs.pop("point_size", 1),
            **kwargs,
        )

    def make_dense_memory(self):
        return DenseGeneratedPointMemory(
            height=self.height,
            width=self.width,
            device=torch.device("cpu"),
            K=self.K,
        )

    def test_temporal_mapping(self):
        self.assertEqual(latent_block_to_pixel_span(0, 3), (0, 9))
        self.assertEqual(latent_block_to_pixel_span(3, 3), (9, 21))
        self.assertEqual(latent_block_to_pixel_span(6, 3), (21, 33))
        self.assertEqual(latent_keyframe_indices(0, 3), (0, 4, 8))
        self.assertEqual(latent_keyframe_indices(3, 3), (12, 16, 20))
        self.assertEqual(latent_keyframe_indices(6, 3), (24, 28, 32))
        self.assertEqual(len(latent_keyframe_indices(0, 60)), 60)
        self.assertEqual(latent_keyframe_indices(0, 60)[-1], 236)

    def test_dense_five_second_trajectory_landmarks(self):
        trajectory_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "traj",
            "yaw_0_45_0_45_5s_120.txt",
        )
        with open(trajectory_path) as handle:
            controls = [[float(value) for value in line.split()] for line in handle]
        self.assertEqual([len(line) for line in controls], [120, 120, 120])
        yaw = controls[1]
        self.assertEqual([yaw[index] for index in (0, 39, 78, 116)], [0, 45, 0, 45])
        self.assertEqual(yaw[117:120], [45, 45, 45])

    def test_v2_1_full_length_trajectory_landmarks(self):
        trajectory_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "traj",
            "yaw_0_45_0_45_237.txt",
        )
        with open(trajectory_path) as handle:
            controls = [[float(value) for value in line.split()] for line in handle]
        self.assertEqual([len(line) for line in controls], [237, 237, 237])
        yaw = controls[1]
        self.assertEqual([yaw[index] for index in (0, 79, 158, 236)], [0, 45, 0, 45])
        self.assertTrue(all(yaw[index] < yaw[index + 1] for index in range(0, 79)))
        self.assertTrue(all(yaw[index] > yaw[index + 1] for index in range(79, 158)))
        self.assertTrue(all(yaw[index] < yaw[index + 1] for index in range(158, 236)))

    def test_reference_renderer_returns_aligned_z_buffer_depth(self):
        try:
            from scripts.render_point_cloud import render_batch
        except ModuleNotFoundError as error:
            if error.name in {"open3d", "cv2"}:
                self.skipTest(f"local renderer dependency is missing: {error.name}")
            raise
        K = torch.tensor([
            [2.0, 0.0, 1.5],
            [0.0, 2.0, 1.5],
            [0.0, 0.0, 1.0],
        ])
        image, mask, depth = render_batch(
            torch.tensor([[0.0, 0.0, 2.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.eye(4),
            K,
            width=4,
            height=4,
            point_size=2,
            ss_ratio=2.0,
            return_depth=True,
        )
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(depth.shape, (4, 4))
        self.assertTrue((depth[mask > 0] == 2.0).all())
        self.assertTrue((depth[mask == 0] == 0.0).all())

    def test_identity_pose_round_trip(self):
        memory = self.make_memory()
        memory.update(self.rgb, self.depth, self.pose, self.mask)
        rendered, mask = memory.render(self.pose, self.K)
        self.assertTrue(mask[0].all())
        self.assertLess((rendered[0] - self.rgb).abs().max().item(), 1e-6)

    def test_c2w_translation_direction(self):
        memory = self.make_memory()
        memory.update(self.rgb, self.depth, self.pose, self.mask)
        translated_c2w = torch.eye(4)
        translated_c2w[0, 3] = 0.1
        _, translated_mask = memory.render(translated_c2w, self.K)
        xs = torch.where(translated_mask[0, 0])[1].float()
        self.assertLess(xs.mean().item(), (self.width - 1) / 2.0)

    def test_reference_priority(self):
        ref_rgb = torch.zeros((1, 3, 2, 2))
        hist_rgb = torch.ones_like(ref_rgb)
        ref_mask = torch.tensor([[[[True, False], [True, False]]]])
        hist_mask = torch.ones_like(ref_mask)
        fused_rgb, fused_mask, hist_only = fuse_reference_and_history(
            ref_rgb, ref_mask, hist_rgb, hist_mask
        )
        self.assertTrue(torch.equal(hist_only, ~ref_mask))
        self.assertTrue(torch.equal(fused_rgb[:, :, ref_mask[0, 0]], ref_rgb[:, :, ref_mask[0, 0]]))
        self.assertTrue(fused_mask.all())

    def test_fixed_max_points(self):
        memory = self.make_memory(max_points=100)
        stats = memory.update(self.rgb, self.depth, self.pose, self.mask)
        self.assertEqual(stats["added_pixels"], self.height * self.width)
        self.assertEqual(memory.point_count, 100)

    def test_block_update_adds_all_frames_before_one_compression(self):
        frame_count = 3
        memory = self.make_memory(max_points=5000, voxel_size=0.01)
        stats = memory.update_block(
            self.rgb.unsqueeze(0).repeat(frame_count, 1, 1, 1),
            self.depth.unsqueeze(0).repeat(frame_count, 1, 1),
            self.pose.unsqueeze(0).repeat(frame_count, 1, 1),
            self.mask.unsqueeze(0).repeat(frame_count, 1, 1),
            K=self.K.unsqueeze(0).repeat(frame_count, 1, 1),
        )
        self.assertEqual(stats["added_pixels"], frame_count * self.height * self.width)
        self.assertEqual(memory.num_updates, frame_count)
        self.assertEqual(memory.point_count, self.height * self.width)

    def test_dense_upper_bound_and_append_only_storage(self):
        self.assertEqual(dense_point_count(117, 480, 832), 46_725_120)
        self.assertEqual(dense_point_count(30, 480, 832), 11_980_800)
        self.assertEqual(dense_point_count(237, 480, 832), 94_648_320)
        self.assertEqual(dense_point_count(60, 480, 832), 23_961_600)
        memory = self.make_dense_memory()
        frame_count = 3
        rgb = self.rgb.unsqueeze(0).repeat(frame_count, 1, 1, 1)
        depth = self.depth.unsqueeze(0).repeat(frame_count, 1, 1)
        poses = self.pose.unsqueeze(0).repeat(frame_count, 1, 1)
        all_pixels = self.mask.unsqueeze(0).repeat(frame_count, 1, 1)
        confidence = torch.full_like(depth, 1e-3)
        stats = memory.update_block(
            rgb, depth, poses, all_pixels,
            K=self.K.unsqueeze(0).repeat(frame_count, 1, 1),
            confidence=confidence,
        )
        expected = frame_count * self.height * self.width
        self.assertEqual(stats["added_pixels"], expected)
        self.assertEqual(memory.point_count, expected)
        self.assertEqual(memory.chunk_count, 1)
        self.assertEqual(memory.confidence_chunks[0].dtype, torch.float16)
        self.assertEqual(memory.storage_bytes_for_points(expected), expected * 26)

    @unittest.skipUnless(
        os.environ.get("RUN_DENSE_FULL_SCALE_TEST") == "1" and torch.cuda.is_available(),
        "set RUN_DENSE_FULL_SCALE_TEST=1 on a CUDA machine for the 46M-point allocation test",
    )
    def test_full_scale_117_frame_dense_allocation(self):
        device = torch.device("cuda")
        height, width, frames = 480, 832, 117
        K = torch.tensor([
            [700.0, 0.0, width / 2.0],
            [0.0, 700.0, height / 2.0],
            [0.0, 0.0, 1.0],
        ], device=device)
        memory = DenseGeneratedPointMemory(height, width, device, K)
        rgb = torch.zeros((frames, 3, height, width), device=device)
        depth = torch.ones((frames, height, width), device=device)
        poses = torch.eye(4, device=device).repeat(frames, 1, 1)
        full_reference_mask = torch.ones(
            (frames, height, width), device=device, dtype=torch.bool
        )
        stats = memory.update_block(rgb, depth, poses, full_reference_mask)
        expected = 46_725_120
        self.assertEqual(stats["added_pixels"], expected)
        self.assertEqual(memory.point_count, expected)
        self.assertEqual(memory.chunk_count, 1)
        self.assertTrue(torch.isfinite(memory.point_chunks[0]).all())
        self.assertTrue(torch.isfinite(memory.color_chunks[0]).all())
        self.assertTrue(torch.isfinite(memory.confidence_chunks[0]).all())

    def test_depth_scale_recovery_and_fallback(self):
        reference = torch.full((2, self.height, self.width), 4.0)
        generated = reference / 2.5
        mask = torch.ones_like(reference, dtype=torch.bool)
        scale, stats = calibrate_depth_scale(
            generated, reference, mask, min_overlap=128
        )
        self.assertTrue(stats["scale_reliable"])
        self.assertAlmostEqual(scale, 2.5, places=5)

        fallback, fallback_stats = calibrate_depth_scale(
            generated, reference, torch.zeros_like(mask),
            previous_scale=scale, min_overlap=128,
        )
        self.assertFalse(fallback_stats["scale_reliable"])
        self.assertAlmostEqual(fallback, scale, places=6)

        ema_scale, ema_stats = calibrate_depth_scale(
            reference / 10.0, reference, mask,
            previous_scale=scale, min_overlap=128,
        )
        self.assertTrue(ema_stats["scale_reliable"])
        self.assertAlmostEqual(ema_scale, 5.0, places=4)

        log_ratios = torch.linspace(-1.0, 1.0, reference.numel()).reshape_as(reference)
        noisy_generated = reference / torch.exp(log_ratios)
        noisy_scale, noisy_stats = calibrate_depth_scale(
            noisy_generated, reference, mask,
            previous_scale=ema_scale, min_overlap=128,
        )
        self.assertFalse(noisy_stats["scale_reliable"])
        self.assertGreater(noisy_stats["log_mad"], 0.15)
        self.assertAlmostEqual(noisy_scale, ema_scale, places=6)

    def test_confidence_does_not_filter_dense_points(self):
        memory = self.make_dense_memory()
        rgb = self.rgb.unsqueeze(0)
        depth = self.depth.unsqueeze(0)
        confidence = compute_depth_confidence(
            depth, scale_reliable=False, log_mad=None
        )
        stats = memory.update_block(
            rgb,
            depth,
            self.pose.unsqueeze(0),
            self.mask.unsqueeze(0),
            confidence=confidence,
        )
        self.assertEqual(stats["added_pixels"], self.height * self.width)
        self.assertEqual(memory.point_count, self.height * self.width)

    def test_dense_two_pass_render_rejects_low_confidence_near_outlier(self):
        memory = DenseGeneratedPointMemory(
            height=1,
            width=1,
            device=torch.device("cpu"),
            K=torch.eye(3),
            min_depth=0.01,
        )
        rgb = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
        memory.update_block(
            rgb,
            torch.tensor([[[1.0]]]),
            torch.eye(4).unsqueeze(0),
            torch.ones((1, 1, 1), dtype=torch.bool),
            confidence=torch.tensor([[[0.01]]]),
        )
        true_rgb = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
        memory.update_block(
            true_rgb,
            torch.tensor([[[2.0]]]),
            torch.eye(4).unsqueeze(0),
            torch.ones((1, 1, 1), dtype=torch.bool),
            confidence=torch.tensor([[[1.0]]]),
        )
        rendered, mask, rendered_depth = memory.render_with_depth(
            torch.eye(4), torch.eye(3)
        )
        self.assertTrue(mask.item())
        self.assertAlmostEqual(rendered_depth.item(), 2.0, places=5)
        self.assertGreater(rendered[0, 1, 0, 0].item(), 0.99)

    def test_dense_chunk_streaming_matches_single_chunk(self):
        streaming = self.make_dense_memory()
        single = self.make_dense_memory()
        rgb = self.rgb.unsqueeze(0).repeat(2, 1, 1, 1)
        depth = self.depth.unsqueeze(0).repeat(2, 1, 1)
        poses = self.pose.unsqueeze(0).repeat(2, 1, 1)
        masks = self.mask.unsqueeze(0).repeat(2, 1, 1)
        confidence = torch.ones_like(depth)
        for frame in range(2):
            streaming.update_block(
                rgb[frame:frame + 1], depth[frame:frame + 1],
                poses[frame:frame + 1], masks[frame:frame + 1],
                confidence=confidence[frame:frame + 1],
            )
        single.update_block(rgb, depth, poses, masks, confidence=confidence)
        streaming_result = streaming.render_with_depth(self.pose, self.K)
        single_result = single.render_with_depth(self.pose, self.K)
        for streamed, combined in zip(streaming_result, single_result):
            self.assertTrue(torch.equal(streamed, combined))

    def test_controller_read_generate_write_cycle(self):
        try:
            from pipeline.historical_memory_controller import HistoricalMemoryController
        except ModuleNotFoundError as error:
            if error.name == "einops":
                self.skipTest("local lightweight torch environment does not include einops")
            raise

        num_frames = 21
        reference_rgb = torch.zeros((1, 3, num_frames, self.height, self.width))
        reference_mask = torch.ones_like(reference_rgb)
        reference_mask[:, :, :, :, self.width // 2:] = -1
        target_c2w = self.pose.repeat(num_frames, 1, 1).unsqueeze(0)
        memory = self.make_memory(max_points=5000)

        def fake_encode(video):
            latent_frames = (video.shape[2] + 3) // 4
            return torch.zeros((1, latent_frames, 16, self.height // 8, self.width // 8))

        with tempfile.TemporaryDirectory() as output_dir:
            controller = HistoricalMemoryController(
                reference_rgb_bcthw=reference_rgb,
                reference_mask_bcthw=reference_mask,
                target_c2w=target_c2w,
                K=self.K.unsqueeze(0),
                encode_video=fake_encode,
                block_decoder=FakeBlockDecoder(self.height, self.width),
                depth_estimator=FakeDepthEstimator(),
                memory=memory,
                output_dir=output_dir,
                output_prefix="0",
                rank=0,
                reference_map_path="reference/frames_pcd",
                save_diagnostics=False,
            )
            render0, mask0 = controller.condition_provider(0, 0, 3)
            self.assertEqual(tuple(render0.shape[:3]), (1, 3, 16))
            self.assertEqual(tuple(mask0.shape[:3]), (1, 3, 4))
            controller.output_callback(
                block_index=0,
                latent_start=0,
                denoised_latent=torch.zeros((1, 3, 16, 3, 4)),
                dit_ms=1.0,
            )
            controller.condition_provider(1, 3, 3)
            self.assertGreater(controller.metrics[1]["hist_only_coverage"], 0)
            controller.output_callback(
                block_index=1,
                latent_start=3,
                denoised_latent=torch.zeros((1, 3, 16, 3, 4)),
                dit_ms=1.0,
            )
            summary = controller.close()
            self.assertEqual(summary["output_frames"], 21)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "0-pred_video_rank0.mp4")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "0-memory_timing_rank0.json")))

    def test_controller_full_block_da3_updates_every_generated_frame(self):
        try:
            from pipeline.historical_memory_controller import HistoricalMemoryController
        except ModuleNotFoundError as error:
            if error.name == "einops":
                self.skipTest("local lightweight torch environment does not include einops")
            raise

        num_frames = 9
        reference_rgb = torch.zeros((1, 3, num_frames, self.height, self.width))
        reference_mask = torch.ones_like(reference_rgb)
        reference_mask[:, :, :, :, self.width // 2:] = -1
        target_c2w = self.pose.repeat(num_frames, 1, 1).unsqueeze(0)
        memory = self.make_memory(max_points=5000)
        depth_estimator = FakeDepthEstimator()

        def fake_encode(video):
            latent_frames = (video.shape[2] + 3) // 4
            return torch.zeros((1, latent_frames, 16, self.height // 8, self.width // 8))

        with tempfile.TemporaryDirectory() as output_dir:
            controller = HistoricalMemoryController(
                reference_rgb_bcthw=reference_rgb,
                reference_mask_bcthw=reference_mask,
                target_c2w=target_c2w,
                K=self.K.unsqueeze(0),
                encode_video=fake_encode,
                block_decoder=FakeBlockDecoder(self.height, self.width),
                depth_estimator=depth_estimator,
                memory=memory,
                output_dir=output_dir,
                output_prefix="0",
                rank=0,
                reference_map_path="reference/frames_pcd",
                memory_update_mode="full_block",
                save_diagnostics=False,
            )
            controller.condition_provider(0, 0, 3)
            controller.output_callback(
                block_index=0,
                latent_start=0,
                denoised_latent=torch.zeros((1, 3, 16, 3, 4)),
                dit_ms=1.0,
            )
            summary = controller.close()
            self.assertEqual(depth_estimator.block_calls, 1)
            self.assertEqual(controller.metrics[0]["update_frames"], 9)
            self.assertEqual(
                controller.metrics[0]["added_pixels"],
                9 * self.height * (self.width // 2),
            )
            self.assertEqual(summary["memory_update_mode"], "full_block")
            self.assertGreater(memory.point_count, 0)

    def test_dense_controller_stores_all_pixels_under_full_reference_mask(self):
        try:
            from pipeline.historical_memory_controller import HistoricalMemoryController
        except ModuleNotFoundError as error:
            if error.name == "einops":
                self.skipTest("local lightweight torch environment does not include einops")
            raise

        num_frames = 9
        reference_rgb = torch.zeros((1, 3, num_frames, self.height, self.width))
        reference_mask = torch.ones_like(reference_rgb)
        reference_depth = torch.ones((1, num_frames, self.height, self.width))
        target_c2w = self.pose.repeat(num_frames, 1, 1).unsqueeze(0)
        memory = self.make_dense_memory()

        def fake_encode(video):
            latent_frames = (video.shape[2] + 3) // 4
            return torch.zeros((1, latent_frames, 16, self.height // 8, self.width // 8))

        with tempfile.TemporaryDirectory() as output_dir:
            controller = HistoricalMemoryController(
                reference_rgb_bcthw=reference_rgb,
                reference_mask_bcthw=reference_mask,
                reference_depth_thw=reference_depth,
                target_c2w=target_c2w,
                K=self.K.unsqueeze(0),
                encode_video=fake_encode,
                block_decoder=FakeBlockDecoder(self.height, self.width),
                depth_estimator=FakeDepthEstimator(),
                memory=memory,
                output_dir=output_dir,
                output_prefix="dense",
                rank=0,
                reference_map_path="reference/frames_pcd",
                memory_update_mode="full_block",
                memory_map_mode="dense_two_layer",
                save_diagnostics=False,
            )
            controller.condition_provider(0, 0, 3)
            controller.output_callback(
                block_index=0,
                latent_start=0,
                denoised_latent=torch.zeros((1, 3, 16, 3, 4)),
                dit_ms=1.0,
            )
            summary = controller.close()
            self.assertEqual(memory.point_count, num_frames * self.height * self.width)
            self.assertEqual(controller.metrics[0]["reference_coverage"], 1.0)
            self.assertEqual(summary["memory_map_mode"], "dense_two_layer")
            self.assertTrue(os.path.exists(os.path.join(
                output_dir,
                "dense-historical_memory_final_rank0_manifest.json",
            )))

    def test_dense_controller_writes_one_keyframe_per_latent(self):
        try:
            from pipeline.historical_memory_controller import HistoricalMemoryController
        except ModuleNotFoundError as error:
            if error.name == "einops":
                self.skipTest("local lightweight torch environment does not include einops")
            raise

        num_frames = 21
        reference_rgb = torch.zeros((1, 3, num_frames, self.height, self.width))
        reference_mask = torch.ones_like(reference_rgb)
        reference_depth = torch.ones((1, num_frames, self.height, self.width))
        target_c2w = self.pose.repeat(num_frames, 1, 1).unsqueeze(0)
        memory = self.make_dense_memory()

        def fake_encode(video):
            latent_frames = (video.shape[2] + 3) // 4
            return torch.zeros((1, latent_frames, 16, self.height // 8, self.width // 8))

        with tempfile.TemporaryDirectory() as output_dir:
            controller = HistoricalMemoryController(
                reference_rgb_bcthw=reference_rgb,
                reference_mask_bcthw=reference_mask,
                reference_depth_thw=reference_depth,
                target_c2w=target_c2w,
                K=self.K.unsqueeze(0),
                encode_video=fake_encode,
                block_decoder=FakeBlockDecoder(self.height, self.width),
                depth_estimator=FakeDepthEstimator(),
                memory=memory,
                output_dir=output_dir,
                output_prefix="dense-keyframes",
                rank=0,
                reference_map_path="reference/frames_pcd",
                memory_update_mode="latent_keyframe",
                memory_map_mode="dense_two_layer",
                save_diagnostics=False,
            )
            for block_index, latent_start in enumerate((0, 3)):
                controller.condition_provider(block_index, latent_start, 3)
                controller.output_callback(
                    block_index=block_index,
                    latent_start=latent_start,
                    denoised_latent=torch.zeros((1, 3, 16, 3, 4)),
                    dit_ms=1.0,
                )
            summary = controller.close()

            self.assertEqual(controller.metrics[0]["update_frame_indices"], [0, 4, 8])
            self.assertEqual(controller.metrics[1]["update_frame_indices"], [12, 16, 20])
            self.assertEqual(controller.metrics[0]["update_frames"], 3)
            self.assertEqual(controller.metrics[1]["update_frames"], 3)
            self.assertEqual(memory.point_count, 6 * self.height * self.width)
            self.assertEqual(summary["memory_update_mode"], "latent_keyframe")


if __name__ == "__main__":
    unittest.main()
