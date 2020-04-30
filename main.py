from gcn import GCN
from train import run
from data import load_data

if __name__=='__main__':
    # load a data according to input
    data = load_data('cora')

    # create GCN model
    model = GCN(data)

    # run the model niter times
    run(data, model, lr=0.01, weight_decay=5e-4, niter=10)
