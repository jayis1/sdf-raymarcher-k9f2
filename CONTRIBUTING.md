# Contributing to SDF Ray Marcher k9f2

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sdf-raymarcher-k9f2.git
   cd sdf-raymarcher-k9f2
   ```
3. **Set up** the development environment:
   ```bash   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** and ensure:
   - All existing tests pass: `pytest tests/ -v`
   - New features have tests
   - Code follows existing style conventions
   - Docstrings are added for new functions/classes

3. **Commit** with a descriptive message:
   ```bash
   git commit -m "Add: brief description of change"
   ```

4. **Push** and create a Pull Request

## Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- Add type hints for function parameters and return values
- Keep functions focused and small
- Use docstrings for all public functions

## Adding New SDF Primitives

1. Add the SDF function to `src/raymarcher_k9f2/primitives.py`
2. Add it to `PRIMITIVE_MAP` and `PRIMITIVE_PARAMS` in `config.py`
3. Export it from `__init__.py`
4. Add tests in `tests/test_raymarcher.py` or `tests/test_new_features.py`
5. Update the README primitives table

## Adding New Scenes

1. Add the builder function to `scenes.py`
2. Register it in the `SCENES` dict
3. Add a camera preset in `cli.py`'s `make_camera()`
4. Add tests that it builds and renders
5. Update the README

## Adding New Tone Mappers

1. Add the function to `tonemap.py`
2. Register it in `_TONEMAP_FUNCS`
3. Add tests for output range and behavior
4. Update the CLI choices and README

## Reporting Bugs

Please open a GitHub issue with:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Any error messages or screenshots

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] New features have tests
- [ ] Code has docstrings
- [ ] README updated if needed
- [ ] No breaking changes to existing API (or documented if necessary)