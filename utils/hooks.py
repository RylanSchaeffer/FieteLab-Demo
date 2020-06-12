# python libraries are imported first
# add imported libraries to requirements.txt
# libraries should be alphabetized
import json
import numpy as np
import os
import torch


def create_hook_fns_dict(hook_fns_frequencies,
                         start_grad_step,
                         num_grad_steps):

    """

    :param hook_fns_frequencies:    list of 2-tuples. The first element of the tuple
                                    should be an integer of how frequently (measured
                                    in the number of gradient steps) the function should
                                    be called, and the second element should the
                                    function to call.

                                    There are two unique values. 0 indicates that
                                    the function should be called only on the first
                                    gradient step. -1 indicates that the function
                                    should be called only on the last gradient.
                                    step.
    :param start_grad_step:         integer
    :param num_grad_steps:          integer
    :return:                        dict mapping gradient steps to functions to
                                    be called after that gradient step.
    """

    hooks_fn_dict = {}
    for freq, hook_fn in hook_fns_frequencies:

        # decide which step(s) to call hook at
        if freq == 0:
            hook_call_at_grad_steps = [start_grad_step]
        elif freq == -1:
            hook_call_at_grad_steps = [start_grad_step + num_grad_steps - 1]
        else:
            hook_call_at_grad_steps = np.arange(
                start=start_grad_step,
                stop=start_grad_step + num_grad_steps,
                step=freq,
                dtype=np.int)

        # add hook object references to hooks_fn_dict at appropriate gradient steps
        for grad_step in hook_call_at_grad_steps:
            if grad_step not in hooks_fn_dict:
                hooks_fn_dict[grad_step] = []
            hooks_fn_dict[grad_step].append(hook_fn)

    return hooks_fn_dict


def create_hook_fns_train(start_grad_step,
                          num_grad_steps):

    hook_fns_frequencies = [
        (0, hook_log_params),
        (50, hook_print_model_progress),
        (50, hook_save_model),
    ]

    train_fn_hook_dict = create_hook_fns_dict(
        hook_fns_frequencies=hook_fns_frequencies,
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    return train_fn_hook_dict


def hook_log_params(hook_input):

    params_file = os.path.join(
        hook_input['tensorboard_writer'].get_logdir(),
        'params.json')
    with open(params_file, "w") as f:
        f.write(json.dumps(hook_input['params'], indent=4, sort_keys=True))


def hook_print_model_progress(hook_input):

    print('Grad Step: {:5d}\tLoss: {:6.3f}\tMSE: {:6.3f}'.format(
        hook_input['grad_step'],
        hook_input['loss'],
        hook_input['p_mse'],))


def hook_save_model(hook_input):
    model = hook_input['model']
    grad_step = hook_input['grad_step']

    save_dict = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=hook_input['optimizer'].state_dict(),
        centers=hook_input['centers'],
        global_step=grad_step)

    checkpoint_file_path = os.path.join(
        hook_input['tensorboard_writer'].get_logdir(),
        'grad_steps={:06d}.pt'.format(
            hook_input['grad_step']))

    torch.save(
        obj=save_dict,
        f=checkpoint_file_path)
