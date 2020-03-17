from gcn import create_gcn_model
from train import run
from data import load_data


if __name__=='__main__':
    data = load_data('cora')
    model = create_gcn_model(data)
    run(data, model, lr=0.01, weight_decay=5e-4, niter=10, verbose=True)
