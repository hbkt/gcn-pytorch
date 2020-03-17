# gcn-pytorch
This is the pytorch inplementation of Graph Convolutional Network.

Thomas N. Kipf, Max Welling [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

## Usage
```
$ python main.py
```

## Explanation of main.py
```python
from gcn import create_gcn_model
from train import run
from data import load_data

if __name__=='__main__':
    # load a data according to input
    data = load_data('cora')

    # create GCN model
    model = create_gcn_model(data)

    # run the model niter times
    run(data, model, lr=0.01, weight_decay=5e-4, niter=10)
```
