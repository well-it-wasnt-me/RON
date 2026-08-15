# Contributing

DeskBot is intentionally modular. Contributions should preserve the separation
between behavior, interfaces, hardware, and providers.

## Pull request checklist

- [ ] `make lint`
- [ ] `make typecheck`
- [ ] `make test`
- [ ] coverage remains acceptable
- [ ] public APIs have useful docstrings
- [ ] configuration changes are documented
- [ ] new behavior has tests
- [ ] `docs/` is updated when public behavior changes

## Style

- Ruff owns formatting and import ordering.
- Prefer `Protocol` for replaceable boundaries.
- Prefer frozen dataclasses for immutable value objects.
- Avoid `Any` in business logic.
- Keep hardware access behind interfaces.
- Use a module logger through `robot.logging.get_logger`.

## Events

New events belong in `robot.events.events`.

An event should be:

- immutable
- small
- meaningful outside the publisher
- tested independently

## Bug reports

Include:

- what happened
- expected behavior
- reproduction steps
- relevant logs
- `make doctor` output where applicable
- Python/dependency versions

## Documentation

Update the relevant module/reference page whenever a public class, endpoint,
configuration block, provider, hardware backend, or workflow changes.
