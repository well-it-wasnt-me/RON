# Simulation

DeskBot provides a headless simulation driver that composes face rendering,
body pose, and a servo overlay without requiring physical hardware.

Use:

```bash
make simulate
```

or the installed `deskbot-simulate` command.

Simulation is intended for development, visual checks, and CI-friendly
experimentation.

::: robot.simulation
    options:
      show_root_heading: true
      members: true
      show_if_no_docstring: true

## Driver

::: robot.simulation.driver
    options:
      show_root_heading: true
      members: true
      show_if_no_docstring: true

## Overlay

::: robot.simulation.overlay
    options:
      show_root_heading: true
      members: true
      show_if_no_docstring: true
