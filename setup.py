from setuptools import find_packages, setup


setup(
    name="mail-classification-model",
    version="0.1.0",
    description="Advanced email classification model with text and email-risk features.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "joblib>=1.3",
        "numpy>=1.24",
        "pandas>=2.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3",
    ],
    entry_points={"console_scripts": ["mail-classifier=mail_classifier.cli:main"]},
    python_requires=">=3.9",
)
