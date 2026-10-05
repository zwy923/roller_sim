"""The machine as hardware: where every section stands (pure geometry) and its MuJoCo model.

    parts       dimensions and MJCF snippets the sections share (a belt plate, a pulley, a skirt)
    feed_belt   the step-down feed belt and its head
    plough      main belt, plough face, lane, side belt
    station     buffer belt, measuring belt, where the separator docks
    separator   the flip separator (shared with the standalone model in designs/flip_separator/)
    sensors     beams, cameras and the volume scanner as hardware
    layout      derive(cfg): every section placed one after the other -- the dict `d` the rest of the package reads
    assembly    build_xml(cfg, d, blocks): the model; what is what in it; the moving-part clearance sweep

Nothing here moves or decides anything. Geometry needs numpy only; MuJoCo is imported where a model is compiled.
"""
from .layout import derive

__all__ = ['derive']
