from setuptools import setup

PROJECT_NAME = "pdf-reading-order"

setup(
    name=PROJECT_NAME,
    packages=["pdf_reading_order"],
    package_dir={"": "src"},
    version="0.23",
    url="https://github.com/huridocs/pdf-reading-order",
    author="HURIDOCS",
    description="This tool returns the reading order of a PDF"
)
