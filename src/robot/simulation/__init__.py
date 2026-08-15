"""Simulation mode.

The :class:`SimulationDriver` runs the same :class:`FaceAnimator` and
:class:`BodyLanguageEngine` used in hardware mode, but renders the
face to a virtual display and draws a stick-figure visualisation of
the four servos on top.

The virtual display is a :class:`MockDisplay` (in-memory RGB888
buffer). The servo overlay is drawn directly into the face framebuffer
by :class:`ServoOverlay` so the simulation output is a single image
per frame.
"""
