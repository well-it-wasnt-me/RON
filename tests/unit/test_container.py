"""Tests for the dependency container."""

from __future__ import annotations

import pytest

from robot.dependency_container import Container
from robot.errors import DependencyResolutionError


def test_container_resolves_singleton() -> None:
    container = Container()

    class A:
        pass

    a_factory_calls = 0

    def factory() -> A:
        nonlocal a_factory_calls
        a_factory_calls += 1
        return A()

    container.register(A, factory)
    a1 = container.resolve(A)
    a2 = container.resolve(A)
    assert a1 is a2
    assert a_factory_calls == 1


def test_container_resolves_transient() -> None:
    container = Container()

    class A:
        pass

    def factory() -> A:
        return A()

    container.register(A, factory, singleton=False)
    assert container.resolve(A) is not container.resolve(A)


def test_container_unknown_raises() -> None:
    container = Container()

    class A:
        pass

    with pytest.raises(DependencyResolutionError):
        container.resolve(A)


def test_container_register_instance() -> None:
    container = Container()

    class A:
        pass

    a = A()
    container.register_instance(A, a)
    assert container.resolve(A) is a


def test_container_has_and_contains() -> None:
    container = Container()

    class A:
        pass

    container.register(A, A)
    assert container.has(A)
    assert A in container
    assert int not in container
