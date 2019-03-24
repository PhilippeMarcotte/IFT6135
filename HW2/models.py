import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from skopt.utils import Real, Integer

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.
from torch.distributions import Categorical


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


rnn_hp_space = [Real(0.1, 20, name="initial_lr"),
                Integer(100, 500, name="emb_size"),
                Integer(100, 500, name="hidden_size"),
                Integer(2, 5, name="num_layers"),
                Real(0.35, 0.65, name="dp_keep_prob")]


class RNNbase(nn.Module):
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob,
                 recurrent_units):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        recurrent_units: the moduleList of recurrent units to loop through
        """

        super(RNNbase, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.recurrent_units = recurrent_units

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(1 - dp_keep_prob)

        self.init_weights_uniform()

    def init_weights_uniform(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """

        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

        return torch.zeros((self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        logits = []
        for i in range(self.seq_len):
            tokens = inputs[i]
            y, hidden = self.forward_token(tokens, hidden)

            logits.append(y)

        logits = torch.stack(logits)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def forward_token(self, tokens, hidden):
        next_hidden = []
        x = self.dropout(self.embedding(tokens))
        for j, layer in enumerate(self.recurrent_units):
            x = layer(x, hidden[j])
            next_hidden.append(x)
            x = self.dropout(x)
        y = self.fc(x)
        return y, torch.stack(next_hidden)

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        softmax = nn.Softmax()
        samples = []
        next_hidden = hidden
        next_input = input
        for i in range(generated_seq_len):
            y, next_hidden = self.forward_token(next_input, next_hidden)
            argmax_next_input = torch.argmax(softmax(y), dim=1)
            next_input = Categorical(softmax(y)).sample()
            samples.append(next_input)

        return torch.stack(samples)



# Problem 1
class RNN(RNNbase):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        recurrent_units = nn.ModuleList([RecurrentUnit(emb_size, hidden_size)])
        recurrent_units.extend(clones(RecurrentUnit(hidden_size, hidden_size), num_layers - 1))

        super(RNN, self).__init__(emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob,
                                  recurrent_units)

class RecurrentUnit(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(RecurrentUnit, self).__init__()
        self.affine_x = nn.Linear(in_size, hidden_size)
        self.affine_h = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        return self.tanh(self.affine_x(input) + self.affine_h(hidden))

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.hidden_size)
        nn.init.uniform_(self.affine_x.weight, -bound, bound)
        nn.init.uniform_(self.affine_h.weight, -bound, bound)
        nn.init.uniform_(self.affine_h.bias, -bound,  bound)

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Parameter(torch.zeros((input_size, 3 * hidden_size)))
        self.U = nn.Parameter(torch.zeros((hidden_size, 3 * hidden_size)))
        self.bw = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bu = nn.Parameter(torch.zeros(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1/np.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.bw, -bound, bound)
        nn.init.uniform_(self.bu, -bound, bound)

    def forward(self, input, hidden):
        gate_x = torch.addmm(self.bw, input, self.W)
        gate_h = torch.addmm(self.bu, hidden, self.U)

        i_r, i_z, i_n = gate_x.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        h_tilde = torch.tanh(i_n + (r * h_n))

        h = (1 - z) * hidden + z * h_tilde
        return h


# Problem 2
class GRU(RNNbase): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
      """
      emb_size:     The number of units in the input embeddings
      hidden_size:  The number of hidden units per layer
      seq_len:      The length of the input sequences
      vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
      num_layers:   The depth of the stack (i.e. the number of hidden layers at
                    each time-step)
      dp_keep_prob: The probability of *not* dropping out units in the
                    non-recurrent connections.
                    Do not apply dropout on recurrent connections.
      """
      recurrent_units = nn.ModuleList([GRUCell(emb_size, hidden_size)])
      recurrent_units.extend(clones(GRUCell(hidden_size, hidden_size), num_layers - 1))
      super(GRU, self).__init__(emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob,
                                recurrent_units)



# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

transformer_hp_space = []

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.n_units = n_units
        self.heads = n_heads
        self.d_k = self.n_units // self.heads
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

        self.q_linear = nn.Linear(self.n_units, self.n_units)
        self.v_linear = nn.Linear(self.n_units, self.n_units)
        self.k_linear = nn.Linear(self.n_units, self.n_units)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.n_units, self.n_units)

        # Initializing
        self.k = np.sqrt(1 / n_units)

        nn.init.uniform_(self.q_linear.weight, -self.k, self.k)
        nn.init.uniform_(self.v_linear.weight, -self.k, self.k)
        nn.init.uniform_(self.k_linear.weight, -self.k, self.k)
        nn.init.uniform_(self.out.weight, -self.k, self.k)

        nn.init.uniform_(self.q_linear.bias, -self.k, self.k)
        nn.init.uniform_(self.v_linear.bias, -self.k, self.k)
        nn.init.uniform_(self.k_linear.bias, -self.k, self.k)
        nn.init.uniform_(self.out.bias, -self.k, self.k)

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        batch_size = query.size(0)

        # linear transformation and splitting into h heads
        k = self.k_linear(key).view(batch_size, -1, self.heads, self.d_k)
        q = self.q_linear(query).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_linear(value).view(batch_size, -1, self.heads, self.d_k)

        # Rearrange shape to [batch_size, heads, sequence_length, d_k]
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        ####### Attention calculation #######
        # Scaled dot product calculation
        attention_span = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if Mask is not None
        if mask is not None:
            mask = mask.unsqueeze(1)   #Create another dimension to fit the mask
            mask = mask.repeat(1, attention_span.size(1), 1, 1)     #Repeat the mask along that new dimension
            attention_span[mask == 0] = -1e9        #Replace zeros in the attention_span with -1e9

        attention_span = F.softmax(attention_span, dim=-1)

        # Apply dropout
        attention_span = self.dropout(attention_span)

        # Attention
        attention = torch.matmul(attention_span, v)

        # Concatenate back all the heads
        concatenated = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.n_units)

        # Output through final layer
        output = self.out(concatenated)
        return output # size: (batch_size, seq_len, self.n_units)






#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

