# Format
python -m black .
python -m isort .

# Check
python -m pylint --rcfile=pyproject.toml mace tests scripts

# Tests
python -m pytest tests
