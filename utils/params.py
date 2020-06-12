params = {
    'model': {
        'architecture': 'rnn',
        'input_dim': 2,
        'hidden_dim': 50,
        'output_dim': 2,
        'num_layers': 1,
        'nonlinearity': 'tanh',
        'description': 'Single Layer RNN with Dense Readout & Sigmoid'
    },
    'optimizer': {
        'optimizer': 'sgd',
        'lr': 1e-4,
        'kwargs': {},
        'description': 'Vanilla SGD'
    },
    'loss_fn': {
        'loss_fn': 'mse'
    },
    'run': {
        'start_grad_step': 0,
        'num_grad_steps': 100,
        'seed': 1,
    },
    'description': 'You can add a really really really long description here.'
}