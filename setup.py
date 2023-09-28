from setuptools import setup, find_packages

setup(
    name="goalspotter",
    version="1.00",
    packages=find_packages(),
    url="https://github.com/Integration-Alpha/goalspotter_public",
    license="Apache-2.0",
    author="Mohammad Mahdavi",
    author_email="moh.mahdavi.l@gmail.com",
    description="GoalSpotter: A Sustainability Objective Detection System",
    keywords=["Sustainability", "Machine Learning", "Natural Language Processing", "Sustainability Objective Detection"],
    install_requires=open("requirements.txt").read().splitlines(),
    include_package_data=True,
)