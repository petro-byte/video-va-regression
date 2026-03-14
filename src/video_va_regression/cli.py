"""
CLI entry-point wrappers for the video VA regression project.

This module provides thin wrapper functions that forward execution
to the corresponding scripts in the `scripts/` directory.

Rationale:
- keeps `scripts/` as pure executable scripts
- avoids making `scripts/` part of the installed Python API
- enables clean console_scripts entry points

This module does not provide a CLI interface.
"""

# =============================================================================
# Imports
# =============================================================================


import sys
from pathlib import Path


# =============================================================================
# Helper: ensure project root is on sys.path
# =============================================================================


def _ensure_project_root_on_path():
    """
    Ensure that the project root (containing `scripts/`) is on sys.path.

    This is required so that the wrapper can import scripts.* modules
    without turning `scripts/` into an installed package.
    """
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


# =============================================================================
# Preprocessing (CLI-only tools)
# =============================================================================


def create_labels():
    _ensure_project_root_on_path()
    from scripts.tools.preprocessing.create_labels import main
    main()


def create_index():
    _ensure_project_root_on_path()
    from scripts.tools.preprocessing.create_index import main
    main()


def create_normalization_stats():
    _ensure_project_root_on_path()
    from scripts.tools.preprocessing.create_normalization_stats import main
    main()


# =============================================================================
# Feature selection & scheduling (pipeline, config-bound)
# =============================================================================


def create_selection():
    _ensure_project_root_on_path()
    from scripts.pipeline.selection.create_selection import main
    main()


def create_schedule():
    _ensure_project_root_on_path()
    from scripts.pipeline.selection.create_schedule import main
    main()


# =============================================================================
# Training & testing (pipeline, config-bound)
# =============================================================================


def run_training():
    _ensure_project_root_on_path()
    from scripts.pipeline.training.run_training import main
    main()


def run_test():
    _ensure_project_root_on_path()
    from scripts.pipeline.training.run_test import main
    main()


# =============================================================================
# Postprocessing / evaluation (CLI-only tools)
# =============================================================================


def create_report():
    _ensure_project_root_on_path()
    from scripts.tools.postprocessing.create_report import main
    main()


def plot_importances():
    _ensure_project_root_on_path()
    from scripts.tools.postprocessing.plot_importances import main
    main()


def plot_predictions():
    _ensure_project_root_on_path()
    from scripts.tools.postprocessing.plot_predictions import main
    main()


def sample_models():
    _ensure_project_root_on_path()
    from scripts.tools.postprocessing.sample_models import main
    main()
