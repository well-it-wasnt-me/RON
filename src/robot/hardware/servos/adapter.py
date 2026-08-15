"""Adapter that lets the legacy :class:`MockServoBus` satisfy the new
:class:`ServoController` protocol without changing the original class.

The adapter adds the two new protocol members (``backend_name`` and
``close``) while delegating everything else to the wrapped bus. This is
the smallest possible surface change and keeps the existing
:class:`MockServoBus` API stable for every test that imports it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from robot.interfaces.servo import Servo, ServoController
from robot.logging import get_logger

_log = get_logger("hardware.servos.adapter")


class _LegacyBusProto(Protocol):
    """Duck-typed shape of a legacy servo bus (e.g. :class:`MockServoBus`)."""

    def get(self, name: str) -> Servo: ...
    def all(self) -> Iterable[Servo]: ...
    async def release_all(self) -> None: ...


class _BusAdapter:
    """Adapt the legacy :class:`MockServoBus` to the :class:`ServoController` protocol."""

    def __init__(self, bus: _LegacyBusProto, backend_name: str) -> None:
        self._bus = bus
        self._backend_name = backend_name
        self._closed = False

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def get(self, name: str) -> Servo:
        return self._bus.get(name)

    def all(self) -> list[Servo]:
        return list(self._bus.all())

    async def release_all(self) -> None:
        await self._bus.release_all()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _log.info("servo.adapter.closed", backend=self._backend_name)


def wrap_servo_controller(bus: object, *, backend_name: str) -> ServoController:
    """Wrap ``bus`` in a :class:`ServoController`-shaped adapter."""
    return _BusAdapter(bus, backend_name=backend_name)  # type: ignore[arg-type]


__all__ = ["wrap_servo_controller"]
