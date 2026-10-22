"""Processing core for TuftView: no Qt imports anywhere in this package.

    from core import Pipeline, PipelineConfig, VideoFileSource
    pipe = Pipeline(PipelineConfig(backend="unet", checkpoint="runs/.../best.pt"), fps=src.fps)
    result = pipe.process_frame(frame)      # -> FrameResult (tufts, mask, tracks)
"""
from .pipeline import Pipeline, PipelineConfig
from .results import FrameResult, TuftResult
from .sources import CameraSource, FrameSource, VideoFileSource, open_source
from .tracker import (ATTACHED, SEPARATED, UNKNOWN, UNSTEADY, LKPredictor, PointPredictor, StallConfig,
                      Tracker, TrackerConfig)

__all__ = ["Pipeline", "PipelineConfig", "FrameResult", "TuftResult", "FrameSource", "VideoFileSource",
           "CameraSource", "open_source", "Tracker", "TrackerConfig", "StallConfig", "PointPredictor",
           "LKPredictor", "ATTACHED", "UNSTEADY", "SEPARATED", "UNKNOWN"]
