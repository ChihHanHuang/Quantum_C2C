import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("version", "r") as f:
    version = f.read()

setuptools.setup(
    name="quantum_c2c",
    version=version,
    author="Chih Han Huang, Khushwanth Kumar, Sumitra Pundlik, Vardaan Sahgal",
    author_email="",
    description="Quantum C2C.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChihHanHuang/Quantum_C2C",
    packages=setuptools.find_packages(),
    keywords=['data compression', 'deep learning', 'quantum computing', 'qiskit'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
    'qiskit>=0.31.0',
    'qiskit-aer>=0.9.1',
    'qiskit-aqua>=0.9.5',
    'qiskit-ibmq-provider>=0.17.0',
    'qiskit-ignis>=0.6.0',
    'qiskit-terra>=0.18.3',
    'torch>=1.7.1',
    'torchaudio>=0.7.2',
    'torchvision>=0.8.2',
    'numpy>=1.19.5'

    ],
    python_requires='>=3.8',
    entry_points='''
        [console_scripts]
        quantum_c2c=quantum_c2c.quantum_c2c:quantum_c2c
    '''
)
