import setuptools

setuptools.setup(
    name="pythermo",
    version="0.0.1",
    packages=[
        "pythermo",
        "pythermo.clex",
        "pythermo.jobs",
        "pythermo.xtal",
        "pythermo.scripts",
    ],
    entry_points={"console_scripts": ["casm-jobs=pythermo.scripts.casm_jobs:main"]},
    install_requires=[
        "numpy",
        "scipy",
        "pymatgen",
        "thermocore@git+https://github.com/Van-der-Ven-Group/thermocore@main#egg=thermocore",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
    python_requires=">=3.10",
)
