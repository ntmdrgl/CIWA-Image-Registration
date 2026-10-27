"""View a saved SfM point cloud through a registered frame's camera."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_frame(reconstruction, frame):
    with np.load(reconstruction) as data:
        indices = data["frame_indices"]
        rows = np.flatnonzero(indices == frame)
        if not len(rows):
            raise ValueError(f"Frame {frame} is not registered. Available frames: {indices.tolist()}")
        row = rows[0]
        image_path = str(data["image_paths"][row])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read source image to determine camera size: {image_path}")
        height, width = image.shape
        K = data["K"]
        camera = o3d.camera.PinholeCameraParameters()
        camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        # The saved pose is already world-to-camera; do not invert it.
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = data["rotations"][row]
        extrinsic[:3, 3] = data["translations"][row]
        camera.extrinsic = extrinsic

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data["points"])
        cloud.colors = o3d.utility.Vector3dVector(data["colors"] / 255.0)
    return cloud, camera


def visualize(cloud, camera, frame):
    intrinsic = camera.intrinsic
    depth = (np.asarray(cloud.points) @ camera.extrinsic[:3, :3].T
             + camera.extrinsic[:3, 3])[:, 2]
    depth = depth[np.isfinite(depth) & (depth > 0)]
    if not len(depth):
        raise ValueError(f"No points are in front of frame {frame}'s camera.")

    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
    app = gui.Application.instance
    app.initialize()
    window = app.create_window(
        f"SfM point cloud — frame {frame} (R: reset camera)",
        intrinsic.width, intrinsic.height,
    )
    view = gui.SceneWidget()
    view.scene = rendering.Open3DScene(window.renderer)
    view.scene.set_background(np.array([0.05, 0.05, 0.05, 1.0]))
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 3.0
    view.scene.add_geometry("points", cloud, material)
    window.add_child(view)

    def reset_camera():
        view.setup_camera(intrinsic, camera.extrinsic, cloud.get_axis_aligned_bounding_box())
        # Preserve the full intrinsic matrix, including its off-center principal point.
        view.scene.camera.set_projection(
            intrinsic.intrinsic_matrix, max(float(depth.min()) * 0.1, 1e-6),
            float(depth.max()) * 1.1, intrinsic.width, intrinsic.height,
        )
        view.force_redraw()

    def on_layout(context):
        view.frame = window.content_rect
        reset_camera()

    def on_key(event):
        if event.type == gui.KeyEvent.Type.DOWN and event.key == gui.KeyName.R:
            reset_camera()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    window.set_on_layout(on_layout)
    view.set_on_key(on_key)
    reset_camera()
    app.run()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="plant_1", help="Reconstructed dataset name (default: plant_1)")
    parser.add_argument("--frame", type=int, default=0, help="Original frame index (default: 0)")
    parser.add_argument(
        "--reconstruction", type=Path,
        help="Explicit reconstruction file; overrides --dataset",
    )
    args = parser.parse_args()
    reconstruction = args.reconstruction or root / "results" / f"sfm_{args.dataset}" / "reconstruction.npz"
    try:
        cloud, camera = load_frame(reconstruction, args.frame)
        print(f"Viewing {len(cloud.points)} points from frame {args.frame}. Press R to reset the view.")
        visualize(cloud, camera, args.frame)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
