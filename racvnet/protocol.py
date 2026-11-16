"""Fixed training and evaluation settings used for Synapse."""

TRAIN_CASES = tuple(f"case{n:04d}" for n in (5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 28, 30, 31, 33, 34, 37, 39, 40))
TEST_CASES = tuple(f"case{n:04d}" for n in (1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38))
ORGANS = ("aorta", "gallbladder", "left_kidney", "right_kidney", "liver", "pancreas", "spleen", "stomach")
IMAGE_SIZE = 224
TRAIN_SLICES = 2211

STAGE1_SETTINGS = dict(epochs=400, batch_size=8, workers=4, seed=11, lr=1e-4,
                       weight_decay=.05, warmup_epochs=10, min_lr=1e-6, amp=True)
LVRM_SETTINGS = dict(epochs=400, batch_size=2, workers=2, seed=2218,
                     patches_per_patient=8, patch=(32, 128, 128), lr=1e-3,
                     weight_decay=1e-4, warmup=3, min_lr=1e-5, clip=1.,
                     synthetic_clean_probability=.2,
                     synthetic_modes=['remote_fp', 'slice_gap', 'confusion', 'boundary'],
                     synthetic_remote_min_distance=12.,
                     probability_smoothing_sigma=[.35, .65, .65],
                     probability_confidence_range=[.7, .98])

IMAGENET_SHA256 = "80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d"
STAGE1_SHA256 = "bb39b100e4b7e34660cd5fc8e37a5d6eb93a8c29718deeedac9152f514640cb1"
LVRM_SHA256 = "2aa73d4ec6ec74c0b10f10c38e9b2d3d832cbe26bccc51854ce60a00514f866b"
