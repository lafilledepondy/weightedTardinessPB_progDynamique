# Weighted Tardiness Problem

This project contains Python implementations for the weighted tardiness scheduling problem.
It includes:

- dynamic programming methods
- subgradient-based lower-bound methods
- small demo scripts and test cases

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Run

The main entry point is `src/main.py`.

```bash
python3 src/main.py
```

You will be asked to choose between:

- dynamic programming
- subgradient methods

For some options, you will also need to provide a data file path from the project data folders.

## Tests

Run the available checks with:

```bash
python3 src/test_progDyn.py
```

## Data

Example instances are stored in:

- `data_aone/`
- `data_ocsc/`

## Project Structure

### Programmation Dynamique

- `src/progDyn.py` contains the dynamic programming logic.
- `src/readData_progDyn.py` contains the data reading logic for the dynamic programming approach.
- `src/test_progDyn.py` contains simple validation tests.

### Subgradient Methods

- `src/problems.py` contains the subgradient-based problem formulations.
- `src/subgradient.py` contains the subgradient methods.

### Main

- `src/main.py` provides the interactive command-line interface.

# License

MIT License. See [LICENSE](LICENSE) for details.
