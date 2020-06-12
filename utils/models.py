# python libraries are imported first
# add imported libraries to requirements.txt
# libraries should be alphabetized
import torch.nn


class RNNWithReadout(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 num_layers,
                 nonlinearity='tanh'):

        super(RNNWithReadout, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.input_size = input_dim
        self.output_size = output_dim

        # defining the layers
        self.rnn = torch.nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity=nonlinearity)

        # create fully connected readout layer
        self.readout = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        # ensure model parameters have the correct type (double)
        self.double()

    def forward(self, x):
        rnn_output, rnn_last_hidden_state = self.rnn(x)

        linear_output = self.readout(rnn_output.reshape(-1, self.hidden_dim))
        sigmoid_output = self.sigmoid(linear_output)

        forward_output = dict(
            rnn_output=rnn_output,
            rnn_last_hidden_state=rnn_last_hidden_state,
            linear_output=linear_output,
            sigmoid_output=sigmoid_output)

        return forward_output
