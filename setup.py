from glob import glob

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("_fastEDM",
        sorted(glob("src/*.cpp")),
        include_dirs=["src/vendor"],
        define_macros=[("VERSION_INFO", __version__), ("JSON", "true")],
        cxx_std=17
        ),
]

setup(
    name="fastEDM",
    packages=["fastEDM"],
    version=__version__,
    author="Patrick Laub",
    author_email="patrick.laub@gmail.com",
    url="https://github.com/",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["numpy", "pandas", "tqdm", "scipy"],
)
