# Experiments for "NeuPSL: Neural Probabilistic Soft Logic" presented at IJCAI 2023

## Requirements and Setup

These experiments expect that you are running on a POSIX (Linux/Mac) system. The specific application dependencies are as follows:

### Requirements
These experiments expect that you are running on a POSIX (Linux/Mac) system.
The specific application dependencies are as follows:
 - Bash >= 4.0
 - Java >= 7
 - Python >= 3.7

Additionally, Python3 dependencies are provided in `requirements.txt`. To install all Python3 dependencies run:
```
pip3 install -r ./requirements.txt
```

### Data Creation
Before running any experiment data must be fetched and formatted.
To create data for an experiment run the following script:
 - Citation network node classification
 ```
 python3 ./citation/scripts/create-data.py
 ```
 - MNIST-Add1 and MNIST-Add2 with overlap
 ```
 python3 ./mnist-addition/scripts/create-data.py
 ```
 - Visual sudoku puzzle classification
 ```
 ./vspc/scripts/create-data.sh
 ``` 

## Basic Execution

To reproduce a NeuPSL experiment from the IJCAI 2023 paper simply run the following script:
```
./scripts/run.sh <experiment>
```
where `<experiment>` may be one of:
 - `citation`: Citation network node classification
 - `mnist-addition`: MNIST-Add1 and MNIST-Add2 with overlap
 - `vspc`: Visual sudoku puzzle classification
