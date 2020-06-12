# python libraries are imported first
# add imported libraries to requirements.txt
# libraries should be alphabetized
import os
import numpy as np
import torch
import torch.nn
import torch.utils.tensorboard

# your code is imported second
import utils.hooks
import utils.models
import utils.params


# functions are sorted alphabetically & separated by two line spaces
def create_loss_fn(loss_fn_params):

    if loss_fn_params['loss_fn'] == 'mse':
        def loss_fn(target, predicted):
            mse = torch.mean((target - predicted).pow(2))
            return mse
    else:
        raise NotImplementedError

    return loss_fn


# functions are sorted alphabetically & separated by two line spaces
def create_model(model_params):
    model = utils.models.RNNWithReadout(
        input_dim=model_params['input_dim'],
        hidden_dim=model_params['hidden_dim'],
        output_dim=model_params['output_dim'],
        num_layers=model_params['num_layers'],
        nonlinearity=model_params['nonlinearity']
    )
    return model


# functions are sorted alphabetically & separated by two line spaces
def create_optimizer(model,
                     optimizer_params):

    if optimizer_params['optimizer'] == 'sgd':
        optimizer_constructor = torch.optim.SGD
    elif optimizer_params['optimizer'] == 'adam':
        optimizer_constructor = torch.optim.Adam
    elif optimizer_params['optimizer'] == 'rmsprop':
        optimizer_constructor = torch.optim.RMSprop
    else:
        raise NotImplementedError('Unknown optimizer string')

    optimizer = optimizer_constructor(
        params=model.parameters(),
        lr=optimizer_params['lr'],
        **optimizer_params['kwargs'])

    return optimizer


# functions are sorted alphabetically & separated by two line spaces
def create_params():
    params = utils.params.params
    return params


# functions are sorted alphabetically & separated by two line spaces
def create_run_id(params):

    included_params = [
        params['model']['architecture'],
        params['model']['hidden_dim'],
        params['model']['num_layers'],
        params['optimizer']['optimizer'],
        params['optimizer']['lr']
    ]
    separator = '_'
    run_id = separator.join(str(ip) for ip in included_params)
    return run_id


# functions are sorted alphabetically & separated by two line spaces
def create_tensorboard_writer(run_dir):
    tensorboard_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=run_dir)
    return tensorboard_writer


# functions are sorted alphabetically & separated by two line spaces
def set_seeds(seed):
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)


# functions are sorted alphabetically & separated by two line spaces
def setup():

    log_dir = 'runs'
    os.makedirs(log_dir, exist_ok=True)
    params = create_params()
    set_seeds(seed=params['run']['seed'])
    run_id = create_run_id(params=params)
    tensorboard_writer = create_tensorboard_writer(
        run_dir=os.path.join(log_dir, run_id))
    model = create_model(
        model_params=params['model'])
    optimizer = create_optimizer(
        model=model,
        optimizer_params=params['optimizer'])
    loss_fn = create_loss_fn(
        loss_fn_params=params['loss_fn'])
    fn_hook_dict = utils.hooks.create_hook_fns_train(
        start_grad_step=params['run']['start_grad_step'],
        num_grad_steps=params['run']['num_grad_steps'])

    setup_results = dict(
        params=params,
        run_id=run_id,
        tensorboard_writer=tensorboard_writer,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        fn_hook_dict=fn_hook_dict,
    )
    return setup_results