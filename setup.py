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
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7",
)
