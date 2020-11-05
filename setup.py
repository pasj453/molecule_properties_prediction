from setuptools import setup


setup(
    name="Servier Molecule Prediction",
    version="0.1",
    packages=["src"],
    install_requires=[
        "tensorflow == 2.0.0",
        "scikit-learn == 0.23.2",
        "scipy == 1.5.2",
        "numpy == 1.19.2",
        "pandas == 1.1.3",
        "flask == 1.1.2",
        "mol2vec"],
    dependency_links=[
        "git+ssh://git@github.com:samoturk/mol2vec.git"
    ],
    entry_points={
        "console_scripts": [
            "servier = src:main.main"
        ]
    }
)
