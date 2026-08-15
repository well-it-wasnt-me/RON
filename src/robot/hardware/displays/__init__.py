"""Display drivers.

* :class:`MockDisplay` - in-memory; used for tests and the demo CLI.
* :class:`GC9A01Display` - raw SPI + GPIO driver for a 240x240 round
  GC9A01 TFT over SPI (with a swappable SPI transport for testing).
* :class:`CircuitPythonDisplay` - Adafruit displayio + gc9a01a driver
  for the same panel; verified working out of the box on Pi 5.
* :class:`DisplayFactory` - selects the configured backend.
"""

from robot.hardware.displays.factory import DisplayFactory
from robot.hardware.displays.gc9a01 import FakeSpiTransport, GC9A01Display
from robot.hardware.displays.mock_display import MockDisplay

# CircuitPythonDisplay is imported lazily (only on Pi 5 with the
# displayio dependencies installed) to keep the test environment
# lightweight. The factory does the same import.

__all__ = [
    "DisplayFactory",
    "FakeSpiTransport",
    "GC9A01Display",
    "MockDisplay",
]
