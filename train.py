# python libraries are imported first
# add imported libraries to requirements.txt

# your code is imported second
import utils.hooks
import utils.models
import utils.run


def train():
    # functions should be short, no more than 30 lines. This improves
    # readability and reusability.
    # lines should be at most 80 characters. this also improves readability
    # and is necessary for vision-impaired people
    setup_results = utils.run.setup()

    train_model(
        params=setup_results['params'],
        model=setup_results['model'],
        loss_fn=setup_results['loss_fn'],
        optimizer=setup_results['optimizer'],
        fn_hook_dict=setup_results['fn_hook_dict'],
        tensorboard_writer=setup_results['tensorboard_writer'])


def train_model(params,
                model,
                optimizer,
                loss_fn,
                fn_hook_dict,
                tensorboard_writer,
                tensorboard_tag_prefix='train/'):

    # sets the model in training mode.
    model.train()

    # for specified number of gradient steps
    start = params['run']['start_grad_step']
    end = start + params['run']['num_grad_steps']
    for grad_step in range(start, end):

        # clear the tape
        optimizer.zero_grad()

        model_input, target = sample_data()

        # call model
        model_output = model(model_input)

        # compute the loss
        loss = loss_fn(
            predicted=model_output['sigmoid_output'],
            target=target)

        # compute gradient of loss w.r.t. model parameters
        loss.backward()

        # take a gradient step
        optimizer.step()

        # if gradient step is in hook_fns, that means we have functions
        # that we want to call. Construct hook_input and call them.
        if grad_step in fn_hook_dict:

            # hook_input contains everything you might need for analysis &
            # plotting. dump everything in here.
            hook_input = dict(
                params=params,
                loss=loss.item(),
                grad_step=grad_step,
                model=model,
                rnn_output=model_output['rnn_output'].cpu().detach().numpy(),
                rnn_hidden=model_output['rnn_hidden'].cpu().detach().numpy(),
                optimizer=optimizer,
                tensorboard_writer=tensorboard_writer,
                tag_prefix=tensorboard_tag_prefix)

            for hook_fn in fn_hook_dict[grad_step]:
                hook_fn(hook_input)


# allow train() to be imported and used elsewhere
if __name__ == '__main__':
    train()
