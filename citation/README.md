# Citation Network Node Classification

### Data Generation
To generate new train/test/valid splits run the following script:
 ```
 python3 ./scripts/create-data.py
 ```

After generation, specifically for the citation network node classification experiment, the neural model components for NeuPSL are pretrained with node labels before begin trained with NeuPSL.
To run pretraining for the citation network node classification experiment run the following command:
```
python3 ./scripts/setup-networks.py
```