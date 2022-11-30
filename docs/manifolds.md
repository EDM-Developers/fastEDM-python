
This demo runs the Python/C++ code after it has been compiled to Web Assembly and run interactively using PyScript. 
In the future, we can use this method to run an entire EDM analysis based on data uploaded to a website & without having to install any software at all!

!!! tip "Work-in-progress" 
    This is currently in the proof-of-concept phase.
    Please let me know if you're into web tech and would like to help build out this demo!

<script defer src="https://pyscript.net/latest/pyscript.js"></script>

Choose a value of $E$:

<div>
  <!-- Source for slider with current value shown: https://stackoverflow.com/a/18936328 -->
  E: <input type="range" min="2" max="5" value="3" class="slider" id="E-slider" oninput="this.nextElementSibling.value = this.value">
  <output>0</output>
</div>

The manifold created based on the toy dataset (the timeseries of increasing integers, starting at 11):  

<div id="manifold-output" style="text-align: center;"></div>

The summary of calling `edm` on this dataset:

<div id="edm-output" style="text-align: center;"></div>

<py-script src="manifolds.py" />

!!! tip "Be patient" 
    When loading this page, pyscript takes ~5-10 seconds to load, so please be patient. 
