# CIWA-Image-Registration

Install requirements with `pip install -r requirements.txt`

Choose a dataset for reconstruction and viewing (both default to `plant_2`):

```bash
python test_SfM.py --dataset plant_1
python visualize_point_cloud.py --dataset plant_1 --frame 0
```

Datasets are read from `data/eth3d/<dataset>/`; reconstructions are saved to
`results/sfm_<dataset>/`. An explicit dataset folder or `--reconstruction` file
can still be supplied.

Export sequential SIFT match figures and a video:

```bash
python test_SfM.py --dataset plant_1 --no-view --save-matches --match-gap 1 --video-fps 10
```

This saves one PNG for each pair `(i, i + gap)`, a `pairs.csv` index, and
`matches.mp4` under `results/sfm_plant_1/matches_gap_1_frames_70/` for the full
plant_1 sequence. Only RANSAC epipolar inliers are drawn; these do not necessarily
survive pose recovery or triangulation. The figure layout and dimensions stay
fixed for video playback. Use `--match-gap 5` for wider-spaced pairs.
