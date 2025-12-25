from setuptools import setup, find_packages

setup(name="dccnn_project",version="0.1",packages=find_packages(include=["modules", "modules.*", "train", "train.*"]))