from setuptools import setup, find_packages

setup(
    name="video-va-regression",
    version="0.1.0",
    description=(
        "An interpretable LightGBM-based baseline for continuous "
        "valence and arousal estimation on the LIRIS-ACCEDE dataset."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Luka Petrovic",
    author_email="luka.petrovic@tum.de",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "torch",
        "tensorboard",
        "joblib",
        "tqdm",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            # --- Preprocessing (CLI-only tools)
            "video-va-create-labels = video_va_regression.cli:create_labels",
            "video-va-create-index = video_va_regression.cli:create_index",
            "video-va-create-normalization-stats = video_va_regression.cli:create_normalization_stats",

            # --- Feature selection & scheduling (pipeline)
            "video-va-create-selection = video_va_regression.cli:create_selection",
            "video-va-create-schedule = video_va_regression.cli:create_schedule",

            # --- Training & testing (pipeline)
            "video-va-run-training = video_va_regression.cli:run_training",
            "video-va-run-test = video_va_regression.cli:run_test",

            # --- Postprocessing / evaluation (CLI-only tools)
            "video-va-create-report = video_va_regression.cli:create_report",
            "video-va-plot-importances = video_va_regression.cli:plot_importances",
            "video-va-plot-predictions = video_va_regression.cli:plot_predictions",
            "video-va-sample-models = video_va_regression.cli:sample_models",
        ]
    },
    include_package_data=False,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
