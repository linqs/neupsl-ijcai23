# Experiments for "NeuPSL: Neural Probabilistic Soft Logic" presented at IJCAI 2023

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

### Neural Model Pretraining
Specifically for the citation network node classification experiment neural model components for NeuPSL are pretrained with node labels before begin trained with NeuPSL.
To run pretraining for the citation network node classification experiment run the following command:
```
python3 ./citation/scripts/setup-networks.py
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
More specifically, it will run NeuPSL on every data setting created by the corresponding create-data script.
Moreover, `./scripts/run.sh` will call the `./<experiment>/cli/run.sh` file that fetchs the PSL `.jar` file from Maven central and uses it to run NeuPSL. 


## Baseline Experiments
Baseline experiments are also provided in this repository. 
To reproduce baseline results for each experiment run the corresponding run script in the `./<experiment>/other-methods/<baseline>/scripts` directory.

