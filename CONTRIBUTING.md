# Contributing to SDF Ray Marcher k9f2

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/jayis1/sdf-raymarcher-k9f2.git
   cd sdf-raymarcher-k9f2
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Run the test suite:**
   ```bash
   pytest tests/ -v
   ```

3. **Run the CLI:**
   ```bash
   raymarcher-k9f2 --scene demo --width 320 --height 240 --output test.png
   ```

## Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Add docstrings to all public functions and classes
- Keep the pure-Python philosophy — no GPU or C extension dependencies

## Adding New Features

### New SDF Primitives

1. Add the SDF function to `src/raymarcher_k9f2/primitives.py`
2. Add a test in `tests/test_raymarcher.py`
3. Update the README primitive table and `__init__.py` exports

### New Scene Builders

1. Add the scene builder function to `src/raymarcher_k9f2/scenes.py`
2. Register it in the `SCENES` dict
3. Add a CLI option in `cli.py`
4. Add a test that it builds and renders

### New Material Presets

1. Add the factory function to `src/raymarcher_k9f2/material.py`
2. Export it from `__init__.py`
3. Add a test for the preset

## Reporting Issues

- Use GitHub Issues
- Include Python version, OS, and steps to reproduce
- For rendering bugs, include the command and scene name

## Pull Requests

- Keep PRs focused on a single change
- Add tests for new functionality
- Ensure all existing tests pass
- Update documentation (README, docstrings) as needed