import math

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


def lstm_cell(input, hidden, cell, w_ih, w_hh, b_ih, b_hh):
    """
    Proceed calculation of one step of LSTM.
    :param input: Tensor, shape (batch_size, input_size)
    :param hidden: hidden state from previous step, shape (batch_size, hidden_size)

    :param cell: cell state from previous step, shape (batch_size, hidden_size)
    :param w_ih: chunk of weights for process input tensor, shape (4 * hidden_size, input_size)
    :param w_hh: chunk of weights for process hidden state tensor, shape (4 * hidden_size, hidden_size)
    :param b_ih: chunk of biases for process input tensor, shape (4 * hidden_size)
    :param b_hh: chunk of biases for process hidden state tensor, shape (4 * hidden_size)
    
    :return: hidden state and cell state of this step.
    """

    gates = torch.mm(input, w_ih.t()) + torch.mm(hidden, w_hh.t()) + b_ih + b_hh
    
    in_ gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

    in_gate     = torch.sigmoid(in_gate)
    forget_gate = torch.sigmoid(forget_gate)
    cell_gate   = torch.tanh(cell_gate)
    out_gate    = torch.sigmoid(out_gate)

    next_cell   = (forget_gate * cell) + (in_gate * cell_gate)
    next_hidden = out_gate * torch.tanh(cell_gate)

    return next_hidden, next_cell


def st_lstm_cell(input_x, (hidden_s, cell_s), (hidden_t, cell_t), w_ih, w_hh_s, w_hh_t, b_ih, b_hh):
    """
    Proceed calculation of one step of STLSTM.
    :param input_l: input of location embedding, shape (batch_size, input_size)
    :param input_s: input of spatial embedding, shape (batch_size, input_size)
    :param input_q: input of temporal embedding, shape (batch_size, input_size)
    :param hidden: hidden state from previous step, shape (batch_size, hidden_size)
    :param cell: cell state from previous step, shape (batch_size, hidden_size)
    :param w_ih: chunk of weights for process input tensor, shape (4 * hidden_size, input_size)
    :param w_hh: chunk of weights for process hidden state tensor, shape (4 * hidden_size, hidden_size)
    :param w_s: chunk of weights for process input of spatial embedding, shape (3 * hidden_size, input_size)
    :param w_q: chunk of weights for process input of temporal embedding, shape (3 * hidden_size, input_size)
    :param b_ih: chunk of biases for process input tensor, shape (4 * hidden_size)
    :param b_hh: chunk of biases for process hidden state tensor, shape (4 * hidden_size)
    :return: hidden state and cell state of this step.
    """
    gates = torch.mm(input_x, w_ih.t()) + torch.mm(hidden_s, w_hh_s.t()) + torch.mm(hidden_t, w_hh_t.t()) + b_ih + b_hh  # Shape (batch_size, 5 * hidden_size)
    
    in_gate, forget_gate_s, forget_gate_t, cell_gate, out_gate = gates.chunk(5, 1)

    in_gate       = torch.sigmoid(in_gate)
    forget_gate_s = torch.sigmoid(forget_gate_s)
    forget_gate_t = torch.sigmoid(forget_gate_t)
    out_gate      = torch.sigmoid(out_gate)
    cell_gate     = torch.tanh(cell_gate)
    

    next_cell   = (in_gate * cell_gate) + (forget_gate_s * cell_s) + (forget_gate_t * cell_t) 
    next_hidden = out_gate * torch.tanh(next_cell)

    return next_hidden, next_cell


class STLSTMCell(nn.Module):
    """
    A Spatial-Temporal Long Short Term Memory (ST-LSTM) cell.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTMCell(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hc = (torch.randn(3, 20), torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        >>>     hc = st_lstm(input_l[i], input_s[i], input_q[i], hc)
        >>>     output.append(hc[0])
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """

        super(STLSTMCell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_ih   = Parameter(torch.Tensor(5 * hidden_size, input_size))
        self.w_hh_s = Parameter(torch.Tensor(5 * hidden_size, hidden_size))
        self.w_hh_t = Parameter(torch.Tensor(5 * hidden_size, hidden_size))

        if bias:
            self.b_ih = Parameter(torch.Tensor(5 * hidden_size))
            self.b_hh = Parameter(torch.Tensor(5 * hidden_size))
        
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)

        self.reset_parameters()

    def check_forward_input(self, input):
        
        if input.size(1) != self.input_size:
            
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
    
    def init_hidden_value(self, input_x):
        zeros = torch.zeros(input_x.size(0), self.hidden_size, dtype=input_x.dtype, device=input_x.device)
        return (zeros, zeros)
    
    def forward(self, input_x, output_s=None, output_t=None):
        """
        Proceed one step forward propagation of ST-LSTM.
        :param input_l: input of location embedding vector, shape (batch_size, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, input_size)
        :param hc: tuple containing hidden state and cell state of previous step.
        :return: hidden state and cell state of this step.
        """
        self.check_forward_input(input_x)
        
        if output_s is None:
            output_s = init_hidden_value(input_x)
    
        if output_t is None:
            output_t = init_hidden_value(input_x)
                    
        self.check_forward_hidden(input_x, output_s[0], '[0]')
        self.check_forward_hidden(input_x, output_s[1], '[0]')
        self.check_forward_hidden(input_x, output_t[0], '[0]')
        self.check_forward_hidden(input_x, output_t[1], '[0]')


        return st_lstm_cell(input_x, output_s, output_t, 
                    w_ih= self.w_ih, w_hh_s= self.w_hh_s, w_hh_t= self.w_hh_t, 
                    b_ih= self.b_ih, b_hh= self.b_hh)


class STLSTM(nn.Module):
    """
    One layer, batch-first Spatial-Temporal LSTM network.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTM(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hidden_out, cell_out = st_lstm(input_l, input_s, input_q)
    """

    def __init__(self, input_size, hidden_size, num_layers= 1, bias= False):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        
        """
        
        super(STLSTM, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        
        self.bias = bias
        self.cell = STLSTMCell(input_size, hidden_size, bias)

    def check_forward_input(self, input_data):

        if not (input_data.size(-1) == self.input_size):
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input_data.size(-1), self.input_size))

    def forward(self, input_data, output_s=None, output_t=None):
        """
        Proceed forward propagation of ST-LSTM network.
        :param input_l: input of location embedding vector, shape (batch_size, step, input_size)

        :param hc: tuple containing initial hidden state and cell state, optional.

        :return: hidden states and cell states produced by iterate through the steps.
        """
        output_hidden, output_cell = []
        
        self.check_forward_input(input_data)

        _, T, S, _ = input_data.size() 
        
        for t in range(T):
            temp_hidden, temp_cell = []
            
            for s in range(S):
                h_t, c_t = self.cell(input_data)
                temp_hidden.append(h_t) 
                temp_cell.append(c_t)

            output_hidden.append(torch.stack(temp_hidden, 1))
            output_cell.append(torch.stack(temp_cell, 1))
        
        return output_hidden, p

