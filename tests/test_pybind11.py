import fastEDM as m
from pathlib import Path

def test_pybind11():
    assert m.__version__ == 'dev'

    assert m.run_command(2) == 0
    assert m.run_command(2, saveInputs='inputs.json') == 0
    assert Path('inputs.json').exists()
    assert Path('inputs.json').stat().st_size > 0
