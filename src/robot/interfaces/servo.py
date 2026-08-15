"""Servo interfaces.

This module exposes two complementary protocols:

* :class:`Servo` - a single addressable servo channel. It is what the
  :class:`~robot.services.executor.ActionExecutor` talks to in order to move
  one servo.
* :class:`ServoController` - a *bus* that owns several :class:`Servo`
  channels. Different backends (``mock``, ``gpio``, ``pca9685``) all
  implement this protocol so the rest of the application never knows which
  one is in use.

Implementations may use PWM via direct GPIO, a PCA9685 board over I2C, or
a simulator. The only contract is that ``controller.get(name)`` returns a
:class:`Servo` that can be commanded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Servo(Protocol):
    """A single addressable servo channel."""

    @property
    def name(self) -> str:
        """Logical name of the servo (e.g. ``"head_pan"``)."""

    @property
    def angle(self) -> float:
        """Current angle in degrees."""

    async def move_to(self, angle: float, duration_s: float = 0.4) -> None:
        """Move to the target angle over ``duration_s`` seconds."""

    async def release(self) -> None:
        """Stop driving the servo (relax PWM)."""


@runtime_checkable
class ServoController(Protocol):
    """A controller that owns a collection of :class:`Servo` channels.

    A controller is selected at application boot by the
    :class:`~robot.hardware.servos.factory.ServoControllerFactory` based on
    the value of ``config.servo.backend``. The rest of the application only
    ever depends on this interface.
    """

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier (e.g. ``"gpio"``, ``"pca9685"``)."""

    def get(self, name: str) -> Servo:
        """Return the :class:`Servo` registered under ``name``."""

    def all(self) -> list[Servo]:
        """Return every :class:`Servo` known to this controller."""

    async def release_all(self) -> None:
        """Relax every servo (PWM goes to 0)."""

    async def close(self) -> None:
        """Release hardware resources; safe to call multiple times."""


__all__ = ["Servo", "ServoController"]
