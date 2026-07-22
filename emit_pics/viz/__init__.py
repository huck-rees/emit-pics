"""Publication / presentation figure layer.

Kept separate from the data-curation and analysis modules: these read the
pipeline's CSV outputs and render *styled* figures. Nothing here recomputes the
science — restyling never touches the analysis, and regenerating a figure is a
fast read-and-render.

* :mod:`emit_pics.viz.style` — the theme registry (``manuscript`` / ``ppt``),
  applied via a context manager so styling is enforced once, not per figure.
* :mod:`emit_pics.viz.figures` — pure ``(DataFrame) -> Figure`` builders, one per
  publication figure, written once and rendered under whichever theme is active.
"""
