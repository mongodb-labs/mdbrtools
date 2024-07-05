from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="mdbrtools",
    version="0.1.0",
    description="Collection of tools for schema parsing and workload generation used by MongoDB Research",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords="mongodb research tools schema queries workload",
    url="http://github.com/mongodb-labs/mdbrtools",
    author="Thomas Rueckstiess, Alana Huang, Robin Vujanic",
    author_email="thomas+mdbrtools@mongodb.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["pandas", "pymongo", "tqdm"],
    zip_safe=False,
)
