"""Cinematic bokeh package.

We deliberately avoid importing torch-dependent modules at package import
time so that the pure-numpy/opencv pieces (bokeh, refiner, utils, evaluate)
can be used in environments without PyTorch. Use:

    from src.pipeline import BokehPipeline, BokehConfig
"""