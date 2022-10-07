
from js import document
from pyodide.ffi.wrappers import add_event_listener

import micropip
await micropip.install(
    'fastEDM-0.0.1-cp310-cp310-emscripten_3_1_14_wasm32.whl'
)

import fastEDM

# Basic manifold creation [basicManifold]
E = 2
tau = 1
p = 1
t = list(range(1, 11))
x = [10+t_i for t_i in t]

def update_E(*args):
    global E
    E = int(document.getElementById("E-slider").value)
    redraw()

add_event_listener(document.getElementById("E-slider"), "input", update_E)

def redraw():
    manifold = fastEDM.create_manifold(t, x, E=E, tau=tau, p=p)
    mani_str = "<br>".join([str(manifold[i]) for i in range(manifold.shape[0])])
    Element("manifold-output").write(mani_str)

redraw()
