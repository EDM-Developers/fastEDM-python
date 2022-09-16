import fastEDM as m

def test_pybind11():
    #assert m.__version__ == '0.0.1'
    assert m.__version__ == 'dev'
    assert m.add(1, 2) == 3
    assert m.subtract(1, 2) == -1
    print(m.add(1,2))
    print(m.run_command(2))
