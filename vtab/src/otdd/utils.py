import os
import sys
import pickle as pkl
import pdb
import shutil
import logging
import tempfile
import contextlib
from tqdm.autonotebook import tqdm
import joblib

def launch_logger(console_level='warning'):
    ############################### Logging Config #################################
    ## Remove all handlers of root logger object -> needed to override basicConfig above
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO) # Has to be min of all the others

    ## create file handler which logs even debug messages, use random logfile name
    logfile = tempfile.NamedTemporaryFile(prefix="otddlog_", dir='/tmp').name
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)

    ## create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    if console_level == 'warning':
        ch.setLevel(logging.WARNING)
    elif console_level == 'info':
        ch.setLevel(logging.INFO)
    elif console_level == 'error':
        ch.setLevel(logging.ERROR)
    else:
        raise ValueError()
    ## create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    _logger.addHandler(fh)
    _logger.addHandler(ch)
    ################################################################################
    return _logger # TODO: find way to retrieve logfile from logger.handlers[0]

def safedump(d,f):
    try:
        pkl.dump(d, open(f, 'wb'))
    except:
        pdb.set_trace()

def append_to_file(fname, l):
    with open(fname, "a") as f:
        f.write('\t'.join(l) + '\n')

def write_to_file(fname, ls):
    with open(fname, "w") as f:
        for l in ls:
            f.write('\t'.join(l) + '\n')

def delete_if_exists(path, typ='f'):
    if typ == 'f' and os.path.exists(path):
        os.remove(path)
    elif typ == 'd' and os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("Unrecognized path type")


def filter_otdd_args(args):
    otdd_args = AttrDict({})
    flow_args = AttrDict({})


    for k,v in args.items():
        if ('flow' in k) or ('step_size' in k):
            flow_args[k] = v

    # # for k in args:
    # #     v = args[k]
    # #     if k in ['num_workers', 'batch_size', 'maxsize', 'resize', 'print_stats']:
    # #         data_args[k] = v
    # #     elif k.startswith('pretrain'):
    # #         pretrain_args[k.replace('pretrain_','')] = v
    # #     elif k.startswith('transfer'):
    # #         transfer_args[k.replace('transfer_','')] = v
    # #     else:
    # #         pass
    #
    # # Check if any args were specified without pretrain|transfer prefix
    # for k in ['optim', 'lr', 'weight_decay', 'momentum', 'criterion',
    #           'print_freq', 'device']:
    #     for subargs in [pretrain_args, transfer_args]:
    #         if (not k in subargs) and (k in args):
    #             subargs[k] = args[k]

    logger.info('OTDD Args:')
    logger.info(pformat(data_args))
    logger.info('Flow Args:')
    logger.info(pformat(pretrain_args))

    return otdd_args, flow_args

def format_results(reslist):
    formatted = []
    for a in reslist:
        if type(a) == str:
            formatted.append(a)
        elif type(a) == int:
            formatted.append(str(a))
        elif type(a) == float:
            formatted.append('{:8.4f}'.format(a))
    return formatted


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
