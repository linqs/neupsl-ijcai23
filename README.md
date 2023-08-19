# Experiments for "NeuPSL: Neural Probabilistic Soft Logic" presented at IJCAI 2023

This repository covers the experiments for the paper [NeuPSL: Neural Probabilistic Soft Logic](https://linqs.org/publications/#id:pryor-ijcai23) presented at IJCAI 2023.

```
@article{pryor2023ijcai,
    title       = {NeuPSL: Neural Probabilistic Soft Logic},
    author      = {Connor Pryor* and
                   Charles Dickens* and
                   Eriq Augustine and
                   Alon Albalak and
                   William Yang Wang and,
                   Lise Getoor},
    booktitle   = {International Joint Conference on Artificial Intelligence (IJCAI)},
    year        = {2023}
}
```



## Requirements
These experiments expect that you are running on a POSIX (Linux/Mac) system.
The specific application dependencies are as follows:
 - Bash >= 4.0
 - Java >= 7
 - Python >= 3.7

Additionally, specific Python3 dependencies to run the exact splits are provided in `requirements.txt`.
If a different version of tensorflow is desired, please regenerate the data.
To install all Python3 dependencies run:
```
pip3 install -r ./requirements.txt
```

## NeuPSL Experiments
To reproduce a NeuPSL experiment from the IJCAI 2023 paper simply run the following script:
```
./scripts/run.sh <experiment>
```
where `<experiment>` may be one of:
 - `citation`: Citation network node classification
 - `mnist-addition`: MNIST-Add1 and MNIST-Add2 with overlap
 - `vspc`: Visual sudoku puzzle classification

The `./scripts/run.sh` script will run NeuPSL on the specified experiment. 
More specifically, it will run NeuPSL on every data setting used in the paper.
To do this it will download the data if it does not exist.
Moreover, `./scripts/run.sh` will call the `./<experiment>/cli/run.sh` file that fetchs the PSL `.jar` file from Maven central and uses it to run NeuPSL. 

For individual experiments or to generate new data, please see the README in the corresponding experiment directory.

## Baseline Experiments
Baseline experiments are also provided in this repository. 
To reproduce baseline results for each experiment run the corresponding run script in the `./<experiment>/other-methods/<baseline>/scripts` directory.

