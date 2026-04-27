from setuptools import setup, find_packages

setup(
    name="synthendosim",
    version="0.1.0",
    description="SynthEndoSim — Synthetic Endovascular Simulation Engine",
    author="Liu Yue",
    author_email="miracle.techlink@gmail.com",
    packages=find_packages(),
    package_data={"synthendosim": ["config/defaults/*.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "gymnasium>=0.29",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "trimesh>=4.0",
    ],
    extras_require={
        "rl": [
            "stable-baselines3>=2.0",
            "tensorboard",
        ],
        "imaging": [
            "pillow",
            "opencv-python",
        ],
    },
)
