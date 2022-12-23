from fastEDM import _fastEDM as m
from pathlib import Path
import unittest


class TestPyBind(unittest.TestCase):
    def test_pybind11(self):
        t = list(range(10))
        x = list(range(10))

        assert (
            m.run_command(
                t, x, es=[2, 3], tau=2, thetas=[2.0], saveFinalPredictions=True
            )["rc"]
            == 0
        )

        assert (
            m.run_command(
                t,
                x,
                es=[2, 3],
                tau=1,
                thetas=[2.0],
                saveFinalPredictions=True,
                saveManifolds=True,
            )["rc"]
            == 0
        )

        assert (
            m.run_command(
                t, x, es=[2, 3], tau=2, thetas=[2.0], saveInputs="inputs.json"
            )["rc"]
            == 0
        )
        json = Path("inputs.json")
        assert json.exists()
        assert json.stat().st_size > 0
        json.unlink(missing_ok=True)
