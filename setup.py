from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements", "r") as fh:
    requirements = fh.read().splitlines()

with open("vfv_instance_segment/version", "r") as fh:
    __version__ = fh.read()

setup(
    name="vfv_instance_segment",
    version=__version__,
    description="VirtualFusedVolume instance segmenter",
    packages=find_packages(),
    package_data={"": ["version"]},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
        # "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        # "Operating System :: Os Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    #test_requirements = ["pytest", "mock", "pudb"],
    url="https://github.com/filippocastelli/neuroseg",
    author="Filippo Maria Castelli",
    author_email="castelli@lens.unifi.it",
    entry_points={
        "console_scripts": [
            "segment_vfv = vfv_instance_segment.vfv_segment:main",
        ]
    }
)
