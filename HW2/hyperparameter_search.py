import argparse
from skopt import Optimizer
from skopt.utils import Real
from models import RNN, GRU, rnn_hp_space, transformer_hp_space
from models import make_model as TRANSFORMER
ptblm = __import__("ptb-lm")
import sys
import os
import torch
from torch import nn
import pickle
from datetime import datetime
import time
import numpy as np

hyperparameters_space = [
    Real(10 ** -5, 10 ** -3, "log-uniform", name='optimizer.lr')
]


def save_optimizer(optimizer, path):
    """
    Save an instance of the hyperparameters optimizer as a pickle.
    :param optimizer: the hyperparameters optimizer to save.
    :param path: where to save.
    """
    with open(path, 'wb+') as f:
        pickle.dump(optimizer, f)


def load_optimizer(path):
    """
    Load a pickled hyperparameters optimizer.
    :param path: where the hyperparameters optimizer was saved.
    :return: the loaded hyperparameters optimizer.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def hyperparameters_parsing(hyperparameters, hp_space, args):
    """
    :param hyperparameters: an array of hyperparameters
    to separate in dictionnaries.
    :param hp_space: the initial hyperparameters space,
    used to get hyperparameters name.
    :param args: the configuration for the training
    """
    for i, hp_value in enumerate(hyperparameters):
        key = hp_space[i].name
        args[key] = hp_value

def run_epoch(model, data, loss_fn, optimizer, device, args, is_train=False, lr=1.0):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    if is_train:
        model.train()
    else:
        model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if args["model"] != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    costs = 0.0
    iters = 0
    losses = []

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptblm.ptb_iterator(data, model.batch_size, model.seq_len)):
        if args["model"] == 'TRANSFORMER':
            batch = ptblm.Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
            #print ("outputs.shape", outputs.shape)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = ptblm.repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch
        # and all time-steps of the sequences.
        # For problem 5.3, you will (instead) need to compute the average loss
        #at each time-step separately.
        loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)
        costs += loss.data.item() * model.seq_len
        losses.append(costs)
        iters += model.seq_len
        if args["debug"]:
            print(step, loss)
            break
        if is_train:  # Only update parameters if training
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            if args["optimizer"] == 'ADAM':
                optimizer.step()
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 10) == 10:
                print("step: {}\t"
                      "loss (sum over all examples' seen this epoch): {}\t"
                      "speed (wps): {}".format(step, costs, iters * model.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters), losses


###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

def train(model, args, train_data, valid_data):
    # LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss()
    optimizer = 0
    if args["optimizer"] == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args["initial_lr"])

    # LEARNING RATE SCHEDULE
    lr = args["initial_lr"]
    lr_decay_base = 1 / 1.15
    m_flat_lr = 14.0  # we will not touch lr for the first m_flat_lr epochs

    print("\n########## Running Main Loop ##########################")
    train_ppls = []
    train_losses = []
    val_ppls = []
    val_losses = []
    best_val_so_far = np.inf
    times = []

    # In debug mode, only run one epoch
    if args["debug"]:
        num_epochs = 1
    else:
        num_epochs = args["num_epochs"]

    # MAIN LOOP
    for epoch in range(num_epochs):
        t0 = time.time()
        with open (os.path.join(args["save_dir"], 'log.txt'), 'a') as f_:
            f_.write(str(t0)+ '\n')
        print('\nEPOCH '+str(epoch)+' ------------------')
        if args["optimizer"] == 'SGD_LR_SCHEDULE':
            lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
            lr = lr * lr_decay # decay lr if it is time

        # RUN MODEL ON TRAINING DATA
        train_ppl, train_loss = run_epoch(model, train_data, loss_fn, optimizer, device, args, True, lr)

        # RUN MODEL ON VALIDATION DATA
        val_ppl, val_loss = run_epoch(model, valid_data, loss_fn, optimizer, device, args)


        # SAVE MODEL IF IT'S THE BEST SO FAR
        if val_ppl < best_val_so_far:
            best_val_so_far = val_ppl
            if args["save_best"]:
                print("Saving model parameters to best_params.pt")
                torch.save(model.state_dict(), os.path.join(args["save_dir"], 'best_params.pt'))
            # NOTE ==============================================
            # You will need to load these parameters into the same model
            # for a couple Problems: so that you can compute the gradient
            # of the loss w.r.t. hidden state as required in Problem 5.2
            # and to sample from the the model as required in Problem 5.3
            # We are not asking you to run on the test data, but if you
            # want to look at test performance you would load the saved
            # model and run on the test data with batch_size=1

        # LOC RESULTS
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.extend(train_loss)
        val_losses.extend(val_loss)
        times.append(time.time() - t0)
        log_str = 'epoch: ' + str(epoch) + '\t' \
                + 'train ppl: ' + str(train_ppl) + '\t' \
                + 'val ppl: ' + str(val_ppl)  + '\t' \
                + 'best val: ' + str(best_val_so_far) + '\t' \
                + 'time (s) spent in epoch: ' + str(times[-1])
        print(log_str)
        with open (os.path.join(args["save_dir"], 'log.txt'), 'a') as f_:
            f_.write(log_str+ '\n')

    # SAVE LEARNING CURVES
    lc_path = os.path.join(args["save_dir"], 'learning_curves.npy')
    print('\nDONE\n\nSaving learning curves to '+lc_path)
    np.save(lc_path, {'train_ppls':train_ppls,
                      'val_ppls':val_ppls,
                      'train_losses':train_losses,
                      'val_losses':val_losses})

    return best_val_so_far
    # NOTE ==============================================
    # To load these, run
    # >>> x = np.load(lc_path)[()]
    # You will need these values for plotting learning curves (Problem 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Penn Treebank Language Modeling')

    # Arguments you may need to set to run different experiments in 4.1 & 4.2.
    parser.add_argument('--data', type=str, default='data',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='GRU',
                        help='type of recurrent net (RNN, GRU, TRANSFORMER)')
    parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                        help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
    parser.add_argument('--seq_len', type=int, default=35,
                        help='number of timesteps over which BPTT is performed')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='size of one minibatch')
    parser.add_argument('--initial_lr', type=float, default=20.0,
                        help='initial learning rate')
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='size of hidden layers. IMPORTANT: for the transformer\
                        this must be a multiple of 16.')
    parser.add_argument('--save_best', action='store_true',
                        help='save the model for the best validation performance')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of hidden layers in RNN/GRU, or number of transformer blocks in TRANSFORMER')

    # Other hyperparameters you may want to tune in your exploration
    parser.add_argument('--emb_size', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs to stop after')
    parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                        help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                        (dp_keep_prob=1 means no dropout)')

    # Arguments that you may want to make use of / implement more code for
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_dir', type=str, default='logs',
                        help='path to save the experimental config, logs, model \
                        This is automatically generated based on the command line \
                        arguments you pass and only needs to be set if you want a \
                        custom dir name')
    parser.add_argument('--evaluate', action='store_true',
                        help="use this flag to run on the test set. Only do this \
                        ONCE for each model setting, and only after you've \
                        completed ALL hyperparameter tuning on the validation set.\
                        Note we are not requiring you to do this.")

    # DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
    # which helps for reproducibility)
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    args = parser.parse_args()
    args = args.__dict__
    ##############################################################################
    #
    # ARG PARSING AND EXPERIMENT SETUP
    #
    ##############################################################################

    args['code_file'] = sys.argv[0]

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args["seed"])

    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")
    # LOAD DATA
    print('Loading data from ' + args["data"])
    raw_data = ptblm.ptb_raw_data(data_path=args["data"])
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    print('  vocabulary size: {}'.format(vocab_size))

    if args["model"] == 'RNN':
        hp_space = rnn_hp_space
    elif args["model"] == 'GRU':
        hp_space = rnn_hp_space
    elif args["model"] == 'TRANSFORMER':
        hp_space = transformer_hp_space
    else:
        raise ValueError("Model type not recognized.")

    hp_optimizer = Optimizer(hp_space)
    hp_search_folder = os.path.join(args["save_dir"], datetime.now().strftime('%m%d_%H%M%S'))
    args["save_dir"] = hp_search_folder
    for i in range(40):
        # Use the model, optimizer, and the flags passed to the script to make the
        # name for the experimental dir
        hyperparameters = hp_optimizer.ask()
        hyperparameters_parsing(hyperparameters, hp_space, args)
        print("\n########## Setting Up Experiment ######################")
        flags = [flag.lstrip('--') for flag in sys.argv[1:]]
        experiment_path = os.path.join(
            args["save_dir"], '_'.join([args['model'],
                                         args[
                                             'optimizer']]
                                        + flags))

        experiment_path = experiment_path + "_" + str(i)

        # Creates an experimental directory and dumps all the args to a text file
        os.makedirs(experiment_path, exist_ok=True)
        print("\nPutting log in %s" % experiment_path)
        args['save_dir'] = experiment_path
        with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
            for key in sorted(args):
                f.write(key + '    ' + str(args[key]) + '\n')

        ###############################################################################
        #
        # MODEL SETUP
        #
        ###############################################################################
        if args["model"] == 'RNN':
            model = RNN(emb_size=args["emb_size"],
                        hidden_size=args["hidden_size"],
                        seq_len=args["seq_len"], batch_size=args["batch_size"],
                        vocab_size=vocab_size, num_layers=args["num_layers"],
                        dp_keep_prob=args["dp_keep_prob"])
        elif args["model"] == 'GRU':
            model = GRU(emb_size=args["emb_size"],
                        hidden_size=args["hidden_size"],
                        seq_len=args["seq_len"], batch_size=args["batch_size"],
                        vocab_size=vocab_size, num_layers=args["num_layers"],
                        dp_keep_prob=args["dp_keep_prob"])
        elif args["model"] == 'TRANSFORMER':
            if args["debug"]:  # use a very small model
                model = TRANSFORMER(vocab_size=vocab_size, n_units=16,
                                    n_blocks=2)
            else:
                # Note that we're using num_layers and hidden_size to mean slightly
                # different things here than in the RNNs.
                # Also, the Transformer also has other hyperparameters
                # (such as the number of attention heads) which can change it's behavior.
                model = TRANSFORMER(vocab_size=vocab_size,
                                    n_units=args["hidden_size"],
                                    n_blocks=args["num_layers"],
                                    dropout=1. - args["dp_keep_prob"])
            # these 3 attributes don't affect the Transformer's computations;
            # they are only used in run_epoch
            model.batch_size = args["batch_size"]
            model.seq_len = args["seq_len"]
            model.vocab_size = vocab_size
        else:
            raise ValueError("Model type not recognized.")

        model = model.to(device)

        best_valid_loss = train(model, args, train_data, valid_data)

        hp_optimizer.tell(hyperparameters, best_valid_loss)

        save_optimizer(hp_optimizer, hp_search_folder)
