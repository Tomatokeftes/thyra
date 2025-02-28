from setuptools import setup, find_packages

setup(
    name="msiconvert",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "zarr",
        "dask",
        "scipy",
        "pyimzml",
        "tqdm",
        # For SpatialData converter
        "spatialdata",
        "anndata",
        "geopandas",
        "shapely",
    ],
    entry_points={
        "console_scripts": [
            "msiconvert=msiconvert.__main__:main",
        ],
    },
)