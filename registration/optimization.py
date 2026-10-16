from mealpy import FloatVar, GWO, WOA
import numpy as np
import cv2
import os

from registration.transformations import get_affine_matrix
from registration.metrics import image_mutual_information, normalized_mutual_information

def run_optimization(
    objective_function,
    lower_bounds,
    upper_bounds,
    parameter_names,
    algorithm,
    epoch,
    pop_size,
    seed,
):
    models = {
        "gwo": GWO.OriginalGWO,
        "woa": WOA.OriginalWOA,
    }

    algorithm = algorithm.lower()
    if algorithm not in models:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if not (
        len(lower_bounds)
        == len(upper_bounds)
        == len(parameter_names)
    ):
        raise ValueError("Bounds and parameter names must have equal lengths.")

    problem = {
        "bounds": FloatVar(
            lb=list(lower_bounds),
            ub=list(upper_bounds),
            name="registration_parameters",
        ),
        "obj_func": objective_function,
        "minmax": "max",
    }

    optimizer = models[algorithm](
        epoch=epoch,
        pop_size=pop_size,
    )

    cpu_count = os.cpu_count() or 1

    if cpu_count < 3:
        best = optimizer.solve(problem, mode="single", seed=seed)
    else:
        n_workers = min(32, cpu_count, pop_size)

        print(f"Using {n_workers} workers on {cpu_count} logical CPUs")

        best = optimizer.solve(
            problem,
            mode="thread",
            n_workers=n_workers,
            seed=seed,
        )

    parameters = {
        name: float(value)
        for name, value in zip(parameter_names, best.solution)
    }

    return parameters, float(best.target.fitness)

def create_dataset_objective_function(
    src_images,
    tgt_images,
    camera_matrix,
    bins,
    metric,
    use_distortion,
):
    """
    Evaluate shared registration parameters using the mean selected metric.
    """
    metrics = {
        "mutual_information": image_mutual_information,
        "normalized_mutual_information": normalized_mutual_information,
    }

    if metric not in metrics:
        raise ValueError(f"Unsupported metric: {metric}")

    score_images = metrics[metric]

    if len(src_images) != len(tgt_images):
        raise ValueError(
            "src_images and tgt_images must contain the same "
            f"number of images, but received {len(src_images)} "
            f"and {len(tgt_images)}."
        )

    if len(src_images) == 0:
        raise ValueError("The dataset cannot be empty.")

    # All images should use consistent dimensions when one shared
    # parameter set is applied.
    src_shape = src_images[0].shape[:2]
    tgt_shape = tgt_images[0].shape[:2]

    for index, (src_image, tgt_image) in enumerate(
        zip(src_images, tgt_images)
    ):
        if src_image.shape[:2] != src_shape:
            raise ValueError(
                f"Source image {index} has shape "
                f"{src_image.shape[:2]}, expected {src_shape}."
            )

        if tgt_image.shape[:2] != tgt_shape:
            raise ValueError(
                f"Target image {index} has shape "
                f"{tgt_image.shape[:2]}, expected {tgt_shape}."
            )

    src_H, src_W = src_shape
    tgt_H, tgt_W = tgt_shape
    source_mask = np.ones(src_shape, dtype=np.float32)
    # Keep histogram edges fixed across candidates for each image pair.
    histogram_bins = [
        [np.histogram_bin_edges(image[np.isfinite(image)], bins=bins)
         for image in (src_image, tgt_image)]
        for src_image, tgt_image in zip(src_images, tgt_images)
    ]

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    def objective_function(solution):
        (
            angle,
            hx, hy,
            sx, sy,
            tx, ty,
            k1, k2, k3,
            p1, p2,
        ) = solution

        affine_matrix = get_affine_matrix(
            angle=angle,
            hx=hx,
            hy=hy,
            sx=sx,
            sy=sy,
            tx=tx,
            ty=ty,
            cx=cx,
            cy=cy,
        )

        # OpenCV order: k1, k2, p1, p2, k3
        distortion_coefficients = np.array(
            [k1, k2, p1, p2, k3],
            dtype=np.float64,
        )

        # Calculate the undistortion maps once for this candidate
        # solution, then reuse them for every source image.
        map_x, map_y = cv2.initUndistortRectifyMap(
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            R=None,
            newCameraMatrix=camera_matrix,
            size=(src_W, src_H),
            m1type=cv2.CV_32FC1,
        )

        # Transform coverage rather than testing intensity, preserving black content.
        undistorted_mask = cv2.remap(
            source_mask, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = cv2.warpAffine(
            (undistorted_mask >= 1 - 1e-6).astype(np.float32),
            affine_matrix[:2, :], (tgt_W, tgt_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) >= 1 - 1e-6

        if not valid_mask.any():
            return -np.inf

        scores = []

        for src_image, tgt_image, pair_bins in zip(
            src_images,
            tgt_images,
            histogram_bins,
        ):
            undistorted_image = cv2.remap(
                src_image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            transformed_image = cv2.warpAffine(
                undistorted_image,
                affine_matrix[:2, :],
                (tgt_W, tgt_H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            score = score_images(
                transformed_image,
                tgt_image,
                bins=pair_bins,
                mask=valid_mask,
            )

            scores.append(score)

        return float(np.mean(scores))

    return objective_function
