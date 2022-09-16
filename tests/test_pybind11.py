import fastEDM as m
from pathlib import Path

def test_pybind11():
    assert m.__version__ == 'dev'

    assert m.run_command(es=[2, 3], tau=2, thetas=[2.0]) == 0
    assert m.run_command(es=[2, 3], tau=2, thetas=[2.0], saveInputs='inputs.json') == 0
    assert Path('inputs.json').exists()
    assert Path('inputs.json').stat().st_size > 0
