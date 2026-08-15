"""Unit tests for the servo backend abstraction."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests.fakes.servo import (
    FakeGpioServoFactory,
    FakePcaDevice,
    make_fake_pca_factory,
)

from robot.config import (
    GPIOServoConfig,
    GPIOServoMapping,
    PCA9685ServoConfig,
    ServoChannelConfig,
    ServosConfig,
)
from robot.errors import ServoError
from robot.hardware.servos.adapter import wrap_servo_controller
from robot.hardware.servos.factory import ServoControllerFactory
from robot.hardware.servos.gpio_controller import RaspberryPiGPIOServoController
from robot.hardware.servos.mock_servo import MockServo, MockServoBus
from robot.hardware.servos.pca9685_controller import PCA9685ServoController
from robot.interfaces.servo import ServoController


def test_gpio_controller_initialises_servos() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)

    assert sorted(factory.servos) == [12, 13, 18, 19]
    assert controller.backend_name == "gpio"
    assert len(factory.calls) == 4
    assert all(call[1:] == (0.0005, 0.0025, 0.02, 50) for call in factory.calls)


def test_gpio_controller_move_to_maps_angle_to_gpiozero_value() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)

    asyncio.run(controller.get("tilt").move_to(-30.0))
    assert factory.servos[13].value == pytest.approx(-1.0)

    asyncio.run(controller.get("tilt").move_to(0.0))
    assert factory.servos[13].value == pytest.approx(0.0)

    asyncio.run(controller.get("tilt").move_to(30.0))
    assert factory.servos[13].value == pytest.approx(1.0)
    assert controller.get("tilt").angle == 30.0


def test_gpio_controller_ignores_redundant_move_to() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)
    pan = controller.get("pan")

    asyncio.run(pan.move_to(0.0))
    first_value = factory.servos[12].value
    first_writes = factory.servos[12].write_count
    asyncio.run(pan.move_to(0.0))

    assert factory.servos[12].value == first_value
    assert factory.servos[12].write_count == first_writes
    assert controller.get("pan").angle == 0.0


def test_gpio_controller_ignores_tiny_pwm_change() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)
    pan = controller.get("pan")

    asyncio.run(pan.move_to(0.0))
    first_writes = factory.servos[12].write_count
    asyncio.run(pan.move_to(0.4))

    assert factory.servos[12].value == pytest.approx(0.0)
    assert factory.servos[12].write_count == first_writes
    assert pan.angle == pytest.approx(0.4)


def test_gpio_controller_reasserts_after_release() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)
    pan = controller.get("pan")

    asyncio.run(pan.move_to(0.0))
    asyncio.run(pan.release())
    assert factory.servos[12].detached is True

    asyncio.run(pan.move_to(0.0))
    assert factory.servos[12].value == pytest.approx(0.0)
    assert factory.servos[12].detached is False
    assert factory.servos[12].write_count == 2  # type: ignore[unreachable]


def test_gpio_controller_preserves_inversion() -> None:
    factory = FakeGpioServoFactory()
    cfg = GPIOServoConfig(
        channels={
            "wrist": ServoChannelConfig(
                min_angle_deg=0.0,
                max_angle_deg=180.0,
                inverted=True,
                gpio_pin=24,
            )
        }
    )
    controller = RaspberryPiGPIOServoController(cfg, servo_factory=factory)

    asyncio.run(controller.get("wrist").move_to(30.0))

    assert controller.get("wrist").angle == 30.0
    # 30deg is 1/6 into the logical range; inversion makes that 2/3
    # from the minimum, i.e. gpiozero value +1/3.
    assert factory.servos[24].value == pytest.approx(2.0 / 3.0)


def test_gpio_controller_rejects_invalid_angle() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)

    with pytest.raises(ServoError):
        asyncio.run(controller.get("pan").move_to(200.0))


def test_gpio_controller_release_all_detaches_servos() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)

    asyncio.run(controller.release_all())

    assert all(servo.detached for servo in factory.servos.values())


def test_gpio_controller_close_closes_servos() -> None:
    factory = FakeGpioServoFactory()
    controller = RaspberryPiGPIOServoController(GPIOServoConfig(), servo_factory=factory)

    asyncio.run(controller.close())
    asyncio.run(controller.close())

    assert all(servo.closed for servo in factory.servos.values())


def test_gpio_controller_custom_channel_through_gpio_pin() -> None:
    factory = FakeGpioServoFactory()
    cfg = GPIOServoConfig(
        channels={"wrist": ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0, gpio_pin=24)}
    )
    controller = RaspberryPiGPIOServoController(cfg, servo_factory=factory)

    assert 24 in factory.servos
    assert controller.get("wrist").name == "wrist"


def test_gpio_controller_missing_pin_raises() -> None:
    factory = FakeGpioServoFactory()
    cfg = GPIOServoConfig(
        channels={"ankle": ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0)}
    )
    with pytest.raises(ServoError):
        RaspberryPiGPIOServoController(cfg, servo_factory=factory)


def test_pca9685_controller_initialises_with_default_channels() -> None:
    factory, created = make_fake_pca_factory()
    controller = PCA9685ServoController(
        PCA9685ServoConfig(address=0x40, frequency=50), device_factory=factory
    )

    assert controller.backend_name == "pca9685"
    assert len(created) == 1
    assert {servo.name for servo in controller.all()} == {
        "pan",
        "tilt",
        "left_arm",
        "right_arm",
    }


def test_pca9685_controller_move_to_sets_duty() -> None:
    factory, created = make_fake_pca_factory()
    controller = PCA9685ServoController(PCA9685ServoConfig(), device_factory=factory)

    asyncio.run(controller.get("pan").move_to(45.0))

    assert created[0].duties[0] == pytest.approx(0.1, abs=1e-3)
    assert controller.get("pan").angle == 45.0


def test_pca9685_controller_rejects_invalid_angle() -> None:
    factory, _ = make_fake_pca_factory()
    controller = PCA9685ServoController(PCA9685ServoConfig(), device_factory=factory)

    with pytest.raises(ServoError):
        asyncio.run(controller.get("pan").move_to(500.0))


def test_pca9685_controller_close_resets_and_closes_device() -> None:
    factory, created = make_fake_pca_factory()
    controller = PCA9685ServoController(PCA9685ServoConfig(), device_factory=factory)

    asyncio.run(controller.get("pan").move_to(45.0))
    asyncio.run(controller.close())

    assert created[0].duties[0] == 0.0
    assert created[0].closed is True


def test_pca9685_controller_custom_channels() -> None:
    factory, created = make_fake_pca_factory()
    cfg = PCA9685ServoConfig(
        channels={"wrist": ServoChannelConfig(channel=7, min_angle_deg=0.0, max_angle_deg=180.0)}
    )
    controller = PCA9685ServoController(cfg, device_factory=factory)

    asyncio.run(controller.get("wrist").move_to(0.0))

    assert 7 in created[0].duties


def test_mock_backend_works() -> None:
    cfg = ServosConfig(backend="mock")
    controller = ServoControllerFactory(cfg).build()
    assert isinstance(controller, ServoController)
    assert controller.backend_name == "mock"
    pan = controller.get("pan")
    assert pan.name == "pan"
    asyncio.run(pan.move_to(30.0))
    assert pan.angle == 30.0
    asyncio.run(controller.close())


def test_factory_dispatches_to_gpio() -> None:
    factory = FakeGpioServoFactory()
    cfg = ServosConfig(backend="gpio")
    controller = ServoControllerFactory(cfg, servo_factory=factory).build()
    assert isinstance(controller, RaspberryPiGPIOServoController)
    assert controller.backend_name == "gpio"


def test_factory_dispatches_to_pca9685() -> None:
    factory, _ = make_fake_pca_factory()
    cfg = ServosConfig(backend="pca9685")
    controller = ServoControllerFactory(cfg, pca_device_factory=factory).build()
    assert isinstance(controller, PCA9685ServoController)
    assert controller.backend_name == "pca9685"


def test_factory_fails_fast_on_unavailable_gpio() -> None:
    def broken_factory(pin: int, channel: ServoChannelConfig, frequency: int) -> None:
        raise ServoError("gpiozero not available")

    cfg = ServosConfig(backend="gpio")
    with pytest.raises(ServoError):
        ServoControllerFactory(cfg, servo_factory=broken_factory).build()  # type: ignore[arg-type]


def test_factory_fails_fast_on_unavailable_pca9685() -> None:
    def broken_factory(config: object) -> FakePcaDevice:
        raise ServoError("PCA9685 SDK not available")

    cfg = ServosConfig(backend="pca9685")
    with pytest.raises(ServoError):
        ServoControllerFactory(cfg, pca_device_factory=broken_factory).build()


def test_config_defaults() -> None:
    cfg = ServosConfig()
    assert cfg.backend == "mock"
    assert cfg.gpio.frequency == 50
    assert cfg.gpio.pins.pan == 12
    assert cfg.gpio.pins.tilt == 13
    assert cfg.gpio.pins.left_arm == 18
    assert cfg.gpio.pins.right_arm == 19
    assert cfg.pca9685.address == 0x40
    assert cfg.pca9685.frequency == 50


def test_config_backend_is_literal() -> None:
    for backend in ("mock", "gpio", "pca9685"):
        cfg = ServosConfig(backend=backend)
        assert cfg.backend == backend


def test_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValidationError):
        ServosConfig(backend="banana")


def test_config_rejects_out_of_range_gpio_pin() -> None:
    with pytest.raises(ValidationError):
        GPIOServoMapping(pan=99)


def test_config_channels_for_backend() -> None:
    cfg = ServosConfig(
        backend="gpio",
        gpio=GPIOServoConfig(
            channels={
                "wrist": ServoChannelConfig(min_angle_deg=0.0, max_angle_deg=180.0, gpio_pin=24)
            }
        ),
    )
    assert "wrist" in cfg.channels_for_backend()


def test_legacy_mock_servos_unaffected() -> None:
    bus = MockServoBus({"a": MockServo(name="a", min_angle=-90, max_angle=90)})
    controller = wrap_servo_controller(bus, backend_name="mock")
    assert isinstance(controller, ServoController)
    assert controller.get("a").name == "a"
    assert bus.get("a").angle == 90.0
