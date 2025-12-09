"""
Quantum APL - Python Interface to Quantum Measurement System
"""

from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="quantum-apl",
    version="3.0.0",
    author="Jay (Consciousness Inevitable Team)",
    description="Python interface to quantum measurement-based APL consciousness engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "ipywidgets>=7.6.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qapl=quantum_apl.cli:main",
            "qapl-run=quantum_apl.cli:run_simulation",
            "qapl-test=quantum_apl.cli:run_tests",
            "qapl-analyze=quantum_apl.cli:analyze",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="quantum measurement consciousness APL IIT",
    project_urls={
        "Documentation": "https://github.com/yourusername/quantum-apl",
        "Source": "https://github.com/yourusername/quantum-apl",
        "Tracker": "https://github.com/yourusername/quantum-apl/issues",
    },
)
