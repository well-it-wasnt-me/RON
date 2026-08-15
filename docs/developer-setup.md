# Developer setup

## Requirements

- Python 3.12+
- `uv`
- POSIX shell for the Makefile workflow
- Real hardware is **not** required for normal development

## Bootstrap

```bash
git clone <your-fork>
cd DeskBot
make install
```

Optional hooks:

```bash
make hooks
```

## Common commands

```bash
make run
make simulate
make doctor

make format
make lint
make typecheck
make test
make coverage
make check

make docs
make docs-serve
```

The normal development stack uses mock hardware.

## Configuration

The root settings model is `robot.config.AppSettings`.

Configuration precedence is designed so environment variables can override
YAML and `.env` values. Nested fields use `__`:

```env
DESKBOT_DISPLAYS__BACKEND=mock
DESKBOT_FACE__THEME=vector
DESKBOT_LLM__PROVIDER=ollama
```

See [Configuration](reference/config.md).

## Testing

Tests are divided into:

- `tests/unit/`
- `tests/integration/`
- `tests/fakes/`

The suite is intended to run without GPIO, SPI, camera, microphone, or speaker
hardware.

## Adding a feature

Prefer this sequence:

1. Define the interface or data contract if the feature crosses a boundary.
2. Implement the behavior independently of hardware.
3. Wire it through the application composition root.
4. Add unit tests.
5. Add an integration test for cross-subsystem behavior.
6. Update documentation and configuration examples.

## Adding a hardware backend

1. Implement the relevant protocol.
2. Add the implementation under `robot.hardware`.
3. Register it in its factory.
4. Add configuration.
5. Add a fake or injectable transport for tests.
6. Document hardware requirements and failure modes.

## Documentation

MkDocs with Material-style extensions and `mkdocstrings` is configured in
`mkdocs.yml`.

Build:

```bash
make docs
```

Serve locally:

```bash
make docs-serve
```
