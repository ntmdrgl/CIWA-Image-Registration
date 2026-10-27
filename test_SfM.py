"""Calibrated incremental SfM for ETH3D RGB sequences.

Reference: https://opencv.org/structure-from-motion-in-opencv/
Camera convention: X_camera = R @ X_world + t. Scale is arbitrary.
"""

import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

MAX_ERROR = 3.0  # pixels
MIN_ANGLE = 1.5  # degrees
MIN_SEED = 30
MIN_PNP = 12


def project(points, pose, K):
    R, t = pose
    camera = np.asarray(points).reshape(-1, 3) @ R.T + t.ravel()
    pixels = camera @ K.T
    depth = camera[:, 2]
    divisor = np.where(np.abs(depth) > 1e-10, depth, 1e-10)
    return pixels[:, :2] / divisor[:, None], depth


def triangulate(uv1, uv2, pose1, pose2, K):
    """Return world points and a mask checking depth, parallax, and error."""
    R1, t1 = pose1
    R2, t2 = pose2
    P1, P2 = (K @ np.column_stack(pose) for pose in (pose1, pose2))
    homogeneous = cv2.triangulatePoints(P1, P2, uv1.T, uv2.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        points = homogeneous[:, :3] / homogeneous[:, 3:]
        pred1, z1 = project(points, pose1, K)
        pred2, z2 = project(points, pose2, K)
        ray1 = points - (-R1.T @ t1).ravel()
        ray2 = points - (-R2.T @ t2).ravel()
        cosine = np.sum(ray1 * ray2, axis=1) / (
            np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1)
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    valid = (
        np.isfinite(points).all(axis=1) & (z1 > 0) & (z2 > 0)
        & (angle >= MIN_ANGLE)
        & (np.linalg.norm(pred1 - uv1, axis=1) <= MAX_ERROR)
        & (np.linalg.norm(pred2 - uv2, axis=1) <= MAX_ERROR)
    )
    return points, valid


def extract_features(paths):
    sift = cv2.SIFT_create(nfeatures=2500)
    features, colors = [], []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        keypoints, descriptors = sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
        )
        uv = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
        xy = np.rint(uv).astype(int)
        xy = np.clip(xy, 0, np.array(image.shape[1::-1]) - 1)
        features.append((uv, descriptors))
        colors.append(image[xy[:, 1], xy[:, 0], ::-1])
    return features, colors


class Reconstruction:
    def __init__(self, K, features, colors):
        self.K, self.features, self.colors = K, features, colors
        self.poses = {}
        self.points, self.point_colors, self.tracks = [], [], []
        self.point_ids = [np.full(len(uv), -1, dtype=int) for uv, _ in features]
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @lru_cache(maxsize=None)
    def match(self, i, j):
        """Mutual SIFT matches passing Lowe's ratio test in both directions."""
        des1, des2 = self.features[i][1], self.features[j][1]
        if des1 is None or des2 is None or min(len(des1), len(des2)) < 2:
            return np.empty((0, 2), dtype=int)

        def ratio(a, b):
            pairs = self.matcher.knnMatch(a, b, k=2)
            return {m.queryIdx: m.trainIdx for pair in pairs if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.75 * n.distance}

        forward, backward = ratio(des1, des2), ratio(des2, des1)
        return np.array([(a, b) for a, b in forward.items()
                         if backward.get(b) == a], dtype=int).reshape(-1, 2)

    def add_point(self, point, i, a, j, b):
        point_id = len(self.points)
        self.points.append(point)
        self.point_colors.append(self.colors[i][a])
        self.tracks.append({i: int(a), j: int(b)})
        self.point_ids[i][a] = self.point_ids[j][b] = point_id

    def initialize(self):
        """Choose a seed pair by the number of geometrically valid points."""
        best = None
        origin = (np.eye(3), np.zeros((3, 1)))
        for i in range(min(3, len(self.features) - 1)):
            for gap in (3, 5, 8, 12, 18, 25, 35):
                j = i + gap
                if j >= len(self.features):
                    continue
                matches = self.match(i, j)
                if len(matches) < MIN_SEED:
                    continue
                uv1 = self.features[i][0][matches[:, 0]]
                uv2 = self.features[j][0][matches[:, 1]]
                E, mask = cv2.findEssentialMat(
                    uv1, uv2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                if E is None or E.shape != (3, 3) or mask is None:
                    continue
                # Filter parallax explicitly instead of using a fixed depth cutoff.
                _, R, t, pose_mask, _ = cv2.recoverPose(
                    E, uv1, uv2, self.K, distanceThresh=1e6, mask=mask.copy()
                )
                points, valid = triangulate(uv1, uv2, origin, (R, t), self.K)
                valid &= pose_mask.ravel() != 0
                count = np.count_nonzero(valid)
                if count >= MIN_SEED and (best is None or count > best[0]):
                    best = count, i, j, R, t, matches[valid], points[valid]
        if best is None:
            raise RuntimeError("No reliable initial pair: need more overlap, texture, or translation.")
        count, i, j, R, t, matches, points = best
        self.seed = (i, j)
        self.poses = {i: origin, j: (R, t)}
        for (a, b), point in zip(matches, points):
            self.add_point(point, i, a, j, b)
        print(f"Seed frames {i}, {j}: {count} points", flush=True)
        return matches

    def register(self, j):
        # Nearby registered views connect new keypoints to existing world points.
        refs = sorted(self.poses, key=lambda i: abs(i - j))[:5]
        correspondences, used_points = {}, set()
        for i in refs:
            for a, b in self.match(i, j):
                point_id = self.point_ids[i][a]
                if point_id >= 0 and b not in correspondences and point_id not in used_points:
                    correspondences[int(b)] = point_id
                    used_points.add(point_id)
        if len(correspondences) < MIN_PNP:
            return False
        keys = np.array(list(correspondences))
        ids = np.array(list(correspondences.values()))
        points = np.array(self.points)[ids]
        uv = self.features[j][0][keys]
        ok, rvec, t, inliers = cv2.solvePnPRansac(
            points, uv, self.K, None, iterationsCount=300,
            reprojectionError=MAX_ERROR, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < MIN_PNP:
            return False
        keep = inliers.ravel()
        rvec, t = cv2.solvePnPRefineLM(points[keep], uv[keep], self.K, None, rvec, t)
        pose = (cv2.Rodrigues(rvec)[0], t)
        predicted, depth = project(points, pose, self.K)
        valid = (depth > 0) & (np.linalg.norm(predicted - uv, axis=1) <= MAX_ERROR)
        if np.count_nonzero(valid) < MIN_PNP:
            return False
        self.poses[j] = pose
        for b, point_id in zip(keys[valid], ids[valid]):
            self.point_ids[j][b] = point_id
            self.tracks[point_id][j] = int(b)

        # Grow the map; each image keypoint belongs to at most one landmark.
        before = len(self.points)
        for i in refs:
            new = [(a, b) for a, b in self.match(i, j)
                   if self.point_ids[i][a] < 0 and self.point_ids[j][b] < 0]
            if not new:
                continue
            pairs = np.array(new)
            uv1 = self.features[i][0][pairs[:, 0]]
            uv2 = self.features[j][0][pairs[:, 1]]
            xyz, good = triangulate(uv1, uv2, self.poses[i], pose, self.K)
            for (a, b), point in zip(pairs[good], xyz[good]):
                self.add_point(point, i, a, j, b)
        print(f"Frame {j}: {valid.sum()} pose inliers, {len(self.points) - before} new points", flush=True)
        return True

    def bundle_adjust(self, max_nfev=40):
        """Sparse robust BA; fix intrinsics, the first pose, and baseline scale."""
        frames = sorted(self.poses)
        frame_index = {frame: k for k, frame in enumerate(frames)}
        ids = [p for p, track in enumerate(self.tracks) if len(track) >= 2]
        if not ids:
            return
        cameras = np.array([np.r_[cv2.Rodrigues(self.poses[i][0])[0].ravel(),
                                  self.poses[i][1].ravel()] for i in frames])
        camera_ids, point_ids, pixels = [], [], []
        for p, point_id in enumerate(ids):
            for frame, key in self.tracks[point_id].items():
                camera_ids.append(frame_index[frame])
                point_ids.append(p)
                pixels.append(self.features[frame][0][key])
        ci, pi, pixels = np.array(camera_ids), np.array(point_ids), np.array(pixels)
        n_camera = cameras.size
        initial = np.r_[cameras.ravel(), np.array(self.points)[ids].ravel()]
        free = np.ones(len(initial), dtype=bool)
        anchor, second = (frame_index[i] for i in self.seed)
        free[6 * anchor:6 * anchor + 6] = False
        # One nonzero translation component removes monocular scale ambiguity.
        free[6 * second + 3 + np.argmax(np.abs(cameras[second, 3:]))] = False

        def unpack(x):
            parameters = initial.copy()
            parameters[free] = x
            return parameters[:n_camera].reshape(-1, 6), parameters[n_camera:].reshape(-1, 3)

        def residual(x):
            cam, xyz = unpack(x)
            rotations = Rotation.from_rotvec(cam[:, :3]).as_matrix()
            camera = np.einsum("nij,nj->ni", rotations[ci], xyz[pi]) + cam[ci, 3:]
            projected = camera @ self.K.T
            z = np.where(np.abs(camera[:, 2:]) > 1e-10, camera[:, 2:], 1e-10)
            return (projected[:, :2] / z - pixels).ravel()

        sparsity = lil_matrix((2 * len(pixels), len(initial)), dtype=int)
        rows = np.arange(len(pixels)) * 2
        for offset in range(6):
            sparsity[rows, ci * 6 + offset] = 1
            sparsity[rows + 1, ci * 6 + offset] = 1
        for offset in range(3):
            sparsity[rows, n_camera + pi * 3 + offset] = 1
            sparsity[rows + 1, n_camera + pi * 3 + offset] = 1
        before = np.sqrt(np.mean(residual(initial[free]) ** 2))
        result = least_squares(
            residual, initial[free], jac_sparsity=sparsity.tocsr()[:, free],
            loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
            ftol=1e-4,
        )
        cam, xyz = unpack(result.x)
        for frame, parameters in zip(frames, cam):
            self.poses[frame] = (cv2.Rodrigues(parameters[:3])[0], parameters[3:].reshape(3, 1))
        for point_id, point in zip(ids, xyz):
            self.points[point_id] = point
        after = np.sqrt(np.mean(residual(result.x) ** 2))
        print(f"Bundle adjustment: {before:.3f} -> {after:.3f} px coordinate RMSE "
              f"({result.nfev} evaluations; {result.message.strip()})", flush=True)
        self.prune()

    def prune(self):
        for point_id, track in enumerate(self.tracks):
            for frame, key in list(track.items()):
                uv, depth = project(self.points[point_id], self.poses[frame], self.K)
                error = np.linalg.norm(uv[0] - self.features[frame][0][key])
                if not np.isfinite(error) or depth[0] <= 0 or error > MAX_ERROR:
                    self.point_ids[frame][key] = -1
                    del track[frame]
            if len(track) < 2:
                for frame, key in track.items():
                    self.point_ids[frame][key] = -1
                track.clear()

    def save(self, output, paths):
        output.mkdir(parents=True, exist_ok=True)
        ids = [p for p, track in enumerate(self.tracks) if len(track) >= 2]
        xyz, colors = np.array(self.points)[ids], np.array(self.point_colors)[ids]
        frames = sorted(self.poses)
        observation_frames, observation_points, observation_pixels = [], [], []
        for p, point_id in enumerate(ids):
            for frame, key in self.tracks[point_id].items():
                observation_frames.append(frame)
                observation_points.append(p)
                observation_pixels.append(self.features[frame][0][key])
        np.savez_compressed(
            output / "reconstruction.npz", points=xyz, colors=colors, K=self.K,
            frame_indices=frames, image_paths=[str(paths[i]) for i in frames],
            rotations=[self.poses[i][0] for i in frames],
            translations=[self.poses[i][1].ravel() for i in frames],
            observation_frames=observation_frames, observation_points=observation_points,
            observation_pixels=observation_pixels,
        )
        with (output / "points.ply").open("w") as file:
            file.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
                       "property float x\nproperty float y\nproperty float z\n"
                       "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            np.savetxt(file, np.column_stack((xyz, colors)), fmt="%.7g %.7g %.7g %d %d %d")
        print(f"Saved {len(frames)}/{len(paths)} cameras and {len(xyz)} points to {output}", flush=True)
        return xyz, colors


def draw_matches(fig, paths, features, pair, matches, title):
    from matplotlib import colormaps
    from matplotlib.patches import ConnectionPatch

    axes = fig.subplots(1, 2)
    colors = colormaps["turbo"](np.linspace(0, 1, len(matches)))
    for column, (ax, frame) in enumerate(zip(axes, pair)):
        image = cv2.imread(str(paths[frame]))
        if image is None:
            raise ValueError(f"Cannot read {paths[frame]}")
        uv = features[frame][0][matches[:, column]]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.scatter(*uv.T, c=colors, s=12)
        ax.set_title(f"Frame {frame}")
        ax.axis("off")
    for (a, b), color in zip(matches, colors):
        fig.add_artist(ConnectionPatch(
            xyA=features[pair[0]][0][a], xyB=features[pair[1]][0][b],
            coordsA="data", coordsB="data", axesA=axes[0], axesB=axes[1],
            color=color, linewidth=0.5, alpha=0.3,
        ))
    fig.suptitle(title)
    # Fixed canvas and margins prevent the image panels jumping between video frames.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.03)


def show_matches(paths, features, seed, matches):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 7))
    draw_matches(fig, paths, features, seed, matches,
                 f"Initial pair: {len(matches)} validated triangulated matches")
    plt.show()


def save_match_sequence(model, paths, output, gap=1, fps=10.0):
    """Save chronological pair diagnostics, including pairs that cannot fit an E.

    These are independent RANSAC checks for (i, i + gap), not a replay of the
    reconstruction's PnP decisions or the map's growth during optimization.
    """
    import csv
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not 1 <= gap < len(paths) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Match gap must be between 1 and N-1; video FPS must be positive.")
    directory = output / f"matches_gap_{gap}_frames_{len(paths)}"
    directory.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(14, 7), dpi=100)
    canvas = FigureCanvasAgg(fig)
    writer = cv2.VideoWriter(str(directory / "matches.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (1400, 700))
    video_open = writer.isOpened()
    if not video_open:
        print("MP4 encoder unavailable; saving the PNG sequence only.", flush=True)
    try:
        with (directory / "pairs.csv").open("w", newline="") as file:
            manifest = csv.writer(file)
            manifest.writerow(["figure", "frame_a", "frame_b", "ratio_matches", "ransac_inliers", "status"])
            for i in range(len(paths) - gap):
                j = i + gap
                matches = model.match(i, j)
                inliers = np.empty((0, 2), dtype=int)
                status = "Not enough matches for RANSAC"
                if len(matches) >= 5:
                    uv1 = model.features[i][0][matches[:, 0]]
                    uv2 = model.features[j][0][matches[:, 1]]
                    E, mask = cv2.findEssentialMat(
                        uv1, uv2, model.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                    )
                    status = "No essential matrix found"
                    if E is not None and mask is not None:
                        inliers = matches[mask.ravel() != 0]
                        status = "RANSAC epipolar inliers (not necessarily valid 3D points)"
                fig.clear()
                title = (f"Frames {i} to {j}: {len(inliers)} / {len(matches)} SIFT matches retained\n"
                         f"{status}")
                draw_matches(fig, paths, model.features, (i, j), inliers, title)
                canvas.draw()
                image = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                name = f"{i:06d}.png"
                if not cv2.imwrite(str(directory / name), image):
                    raise OSError(f"Cannot write {directory / name}")
                if video_open:
                    writer.write(image)
                manifest.writerow([name, i, j, len(matches), len(inliers), status])
    finally:
        writer.release()
        fig.clear()
    print(f"Saved {len(paths) - gap} match figures" + (" and matches.mp4" if video_open else "")
          + f" to {directory}", flush=True)
    return directory


def show_cloud(points, colors, poses):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
    centers = np.array([(-R.T @ t).ravel() for _, (R, t) in sorted(poses.items())])
    path = o3d.geometry.LineSet()
    path.points = o3d.utility.Vector3dVector(centers)
    path.lines = o3d.utility.Vector2iVector(
        np.column_stack((np.arange(len(centers) - 1), np.arange(1, len(centers))))
    )
    path.paint_uniform_color([1.0, 0.7, 0.0])
    o3d.visualization.draw_geometries([cloud, path])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", nargs="?", type=Path, help="Optional dataset folder; overrides --dataset")
    parser.add_argument("--dataset", default="plant_1", help="ETH3D dataset name (default: plant_1)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, help="Use only the first N frames for a quick run")
    parser.add_argument("--no-view", action="store_true", help="Save results without opening Open3D")
    parser.add_argument("--plot-matches", action="store_true", help="Show the validated initial matches")
    parser.add_argument("--save-matches", action="store_true", help="Save sequential RANSAC match PNGs and an MP4")
    parser.add_argument("--match-gap", type=int, default=1, help="Compare frames i and i+gap (default: 1)")
    parser.add_argument("--video-fps", type=float, default=10.0, help="Match video playback FPS (default: 10)")
    args = parser.parse_args()
    dataset_dir = args.dataset_path or root / "data/eth3d" / args.dataset
    paths = sorted((dataset_dir / "rgb").glob("*.png"), key=lambda p: float(p.stem))
    if args.max_frames is not None:
        if args.max_frames < 4:
            parser.error("--max-frames must be at least 4")
        paths = paths[:args.max_frames]
    if len(paths) < 4:
        parser.error(f"Need at least four PNG frames in {dataset_dir / 'rgb'}")
    if args.save_matches and not 1 <= args.match_gap < len(paths):
        parser.error("--match-gap must be between 1 and the number of frames minus one")
    if args.save_matches and (not np.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error("--video-fps must be finite and positive")
    fx, fy, cx, cy = np.loadtxt(dataset_dir / "calibration.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    cv2.setRNGSeed(0)
    print(f"Extracting features from {len(paths)} frames...", flush=True)
    features, colors = extract_features(paths)
    model = Reconstruction(K, features, colors)
    seed_matches = model.initialize()
    model.bundle_adjust()
    pending = set(range(len(paths))) - model.poses.keys()
    # Retry failed frames after later views have expanded the map.
    while pending:
        registered = 0
        for frame in sorted(pending):
            if model.register(frame):
                pending.remove(frame)
                registered += 1
                if len(model.poses) % 10 == 0:
                    model.bundle_adjust(max_nfev=25)
        if not registered:
            break
    model.bundle_adjust(max_nfev=80)
    if pending:
        print(f"Unregistered frames (insufficient reliable correspondences): {sorted(pending)}")
    output = args.output or root / "results" / f"sfm_{dataset_dir.name}"
    points, colors = model.save(output, paths)
    if args.save_matches:
        save_match_sequence(model, paths, output, args.match_gap, args.video_fps)
    if args.plot_matches:
        show_matches(paths, features, model.seed, seed_matches)
    if not args.no_view:
        show_cloud(points, colors, model.poses)


if __name__ == "__main__":
    main()
