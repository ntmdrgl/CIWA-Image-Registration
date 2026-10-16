import numpy as np
from sklearn.metrics import mutual_info_score

def _joint_histogram(image1, image2, bins, mask):
    valid = np.isfinite(image1) & np.isfinite(image2)
    if mask is not None:
        valid &= mask
    return np.histogram2d(image1[valid], image2[valid], bins=bins)[0]


def image_mutual_information(image1, image2, bins, mask=None):
    """Compute MI over finite, valid pixels."""
    joint_hist = _joint_histogram(image1, image2, bins, mask)
    if not joint_hist.any():
        return -np.inf

    return mutual_info_score(
        None,
        None,
        contingency=joint_hist,
    )


def normalized_mutual_information(image1, image2, bins, mask=None):
    """Studholme NMI: (H(A) + H(B)) / H(A, B), normally in [1, 2].

    Return 1 for constant pairs (no information), or -inf for no samples.
    """
    joint_hist = _joint_histogram(image1, image2, bins, mask)
    total = joint_hist.sum()
    if total == 0:
        return -np.inf

    joint = joint_hist / total

    def entropy(probabilities):
        positive = probabilities[probabilities > 0]
        return -np.sum(positive * np.log(positive))

    joint_entropy = entropy(joint)
    if joint_entropy == 0:
        return 1.0

    return float(
        (entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0)))
        / joint_entropy
    )
