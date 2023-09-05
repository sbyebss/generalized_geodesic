import os
import sys
import shutil
from attrdict import AttrDict
from tqdm.autonotebook import tqdm

import time
import numpy as np
import torch
import torch.nn as nn
from pprint import pprint, pformat
from .datasets import load_torchvision_data, load_imagenet
import pdb
from .. import OUTPUT_DIR
import logging
import ignite
import ignite.contrib
from ignite.contrib.handlers.tqdm_logger import ProgressBar


from .utils import process_device_arg

logger = logging.getLogger(__name__)

try:
    from ignite.contrib.metrics.gpu_info import GpuInfo
except:
    logger.warning('GpuInfo not loaded')


OPTIM = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
}

def filter_transfer_args(args):
    data_args  = AttrDict({})
    pretrain_args = AttrDict({})
    transfer_args = AttrDict({'transferring': True})

    for k in args:
        v = args[k]
        if k in ['num_workers', 'batch_size', 'maxsize', 'resize', 'print_stats']:
            data_args[k] = v
        elif k.startswith('pretrain'):
            pretrain_args[k.replace('pretrain_','')] = v
        elif k.startswith('transfer'):
            transfer_args[k.replace('transfer_','')] = v
        else:
            pass

    # Check if any args were specified without pretrain|transfer prefix
    for k in ['optim', 'lr', 'weight_decay', 'momentum', 'criterion',
              'print_freq', 'device']:
        for subargs in [pretrain_args, transfer_args]:
            if (not k in subargs) and (k in args):
                subargs[k] = args[k]

    logger.info('Data Args:')
    logger.info(pformat(data_args))
    logger.info('Pretrain Args:')
    logger.info(pformat(pretrain_args))
    logger.info('Transfer Args:')
    logger.info(pformat(transfer_args))

    return data_args, pretrain_args, transfer_args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def transfer_learn(model_class, src, tgt, args, src_model=None, class_map_layer=None):
    """ Runs simple tranfer learning pipeline

    Arguments:
        src_model (pytorch model, optional):  If provided, will use this model
            instead of pretraining on src data

    """
    ### Process args
    args = default_args(args)
    nopretrain = (type(src) is str and src == 'random') or (src is None)
    imagenet_pretrain = type(src) is str and src == 'ImageNet'
    data_args, pretrain_args, transfer_args = filter_transfer_args(args)

    ### Load Datasets
    # Load src data - only if pretraining on something that is not ImageNet
    if (not nopretrain) and (not imagenet_pretrain):
        if type(src) is str:
            trn_loader_src, val_loader_src, tst_loader_src,_,_ = load_torchvision_data(src, **data_args)
        elif type(src) is tuple:
            trn_loader_src, val_loader_src, tst_loader_src = src
        elif type(src) is dict:
            trn_loader_src, val_loader_src, tst_loader_src = src['train'], src['valid'], src['test']
    else:
        src_acc = None

    # Load tgt data
    if type(tgt) is str:
        trn_loader_tgt, val_loader_tgt, tst_loader_tgt,train_tgt,_ = load_torchvision_data(tgt, **data_args)
    elif type(tgt) is tuple:
        trn_loader_tgt, val_loader_tgt, tst_loader_tgt = tgt
    elif type(tgt) is dict:
        trn_loader_tgt, val_loader_tgt, tst_loader_tgt = tgt['train'], tgt['valid'], tgt['test']
    else:
        raise ValueError('trg should be either name of dataset or tuple of dataloaders')

    ### Train or Pretrain or Randomly-Initialize Source Model
    #pdb.set_trace()
    src_acc = None

    if src_model is not None:
        model = src_model
        #def validate(val_loader, model, criterion, args, prefix = 'Test: '):
        #pretrain_args['compute_accuracy'] = True
        #_, src_acc = validate(tst_loader_src, model, args=pretrain_args)
    elif imagenet_pretrain:
        # No need to train on src - load pretrained (pytorch pretrained models are pretrained on ImageNet)
        model = model_class(pretrained=True)
    elif nopretrain:
        # No pretraining / No transfer
        model = model_class(pretrained=False)
    else:
        model = model_class(pretrained=False)
        #print('Pretraning')
        model, src_acc = train(model, trn_loader_src, val_loader_src, tst_loader_src, args=pretrain_args)

    ### Convert source model
    #print(model)
    # NOTE: All this assumes last layer is always named, and has name 'fc'. Is this always the case for torchvision?
    if class_map_layer is not None:
        # This maps nclasses to nclasses, stack it on top of model
        model = torch.nn.Sequential(model, class_map_layer)
    elif hasattr(model, '_init_classifier'):
        model._init_classifier(num_classes=len(trn_loader_tgt.dataset.classes))
    elif hasattr(model, 'fc'):
        model.fc = torch.nn.Linear(model.fc.in_features, len(trn_loader_tgt.dataset.classes))
    else:
        logger.warning("Don't know how to re-initialize final layer. Will continue without reinit.")
        pdb.set_trace()


    print(model)
    pdb.set_trace()

    if not nopretrain and args.freeze_bottom:
        logger.info('Freezing all but last/classification layer...')
        for n,child in model.named_children():
            if n == 'fc' or n == 'classifier': continue
            for param in child.parameters():
                param.requires_grad = False
        transfer_args.frozen = True

    logger.info('Fine tuning....')
    print(iter(trn_loader_tgt).next()[1].max())

    model, tgt_acc = train(model, trn_loader_tgt, val_loader_tgt, tst_loader_tgt, args=transfer_args)

    return model, src_acc, tgt_acc



def train(model, train_loader, val_loader=None, test_loader = None, args=None):
    args = default_args(args)
    if torch.cuda.is_available() and 'cuda' in args.device:
        #device = torch.device("cuda:{}".format(args.gpu))
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    model = model.to(device)

    if 'frozen' in args and args.frozen:
        # Filter needed to avoid error with frozen layers
        _params_call = lambda m: filter(lambda p: p.requires_grad, m.parameters())
    else:
        _params_call = lambda m: m.parameters()

    if args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(_params_call(model), lr=args.lr, momentum=args.momentum)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(_params_call(model), lr=args.lr)

    if args.criterion == 'crossent':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.criterion == 'mse':
        criterion = nn.MSELoss().to(device)
    else:
        raise ValueError('Criterion not recognized')


    best_loss = np.Inf
    best_acc1 = 0
    is_best   = True

    pbar = tqdm(np.arange(args.epochs)+1, leave=False)
    for epoch in pbar:
        pbar.set_description(f'Training Epoch {epoch}')
        adjust_learning_rate(optimizer, epoch, args)

        train_epoch(train_loader, model, criterion, optimizer, epoch, args)

        if val_loader:
            loss, acc1 = validate(val_loader, model, criterion, args, prefix='Valid: ')

            if args.compute_accuracy: # if acc is computed, use it for selection
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
            else: # fall back to loss for selection
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model.__class__.__name__, # TODO: How to get nlayers and add here?
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_dir)

    ### Reload and return best model
    model,_,_ = load_checkpoint(args.save_dir, model, optimizer, args)

    if test_loader:
        loss, acc1 = validate(test_loader, model, criterion, args)
        out_stat = acc1 if args.compute_accuracy else loss
    else: # Return best validation error instead
        out_stat = best_acc1 if args.compute_accuracy else best_loss
    return model, out_stat



def default_args(args):
    # THere's gotta be a cleaner way of doing this.s
    if type(args) is dict:
        args = AttrDict(args)
    elif not hasattr(args,'items'):
        # Args where probably defined as Class from jupyter notebook
        args = AttrDict({a: args.__getattribute__(a) for a in dir(args) if not a.startswith('__')})
    args.print_freq = 20 if (not hasattr(args, 'print_freq')) else args.print_freq
    args.time_meter = False if not hasattr(args, 'time_meter') else args.time_meter
    args.save_dir   = OUTPUT_DIR if not hasattr(args, 'save_dir') else args.save_dir
    args.compute_accuracy = True if not hasattr(args, 'compute_accuracy') else args.compute_accuracy
    return args

def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    meters = []
    if args.time_meter:
        batch_time = AverageMeter('Time', ':6.3f')
        meters.append(batch_time)
        data_time = AverageMeter('Data', ':6.3f')
        meters.append(data_time) # Data Time

    losses = AverageMeter('Loss', ':.4e')
    meters.append(losses)

    if args.compute_accuracy:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        meters += [top1, top5]

    progress = ProgressMeter(len(train_loader), meters,
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        if args.time_meter: data_time.update(time.time() - end)

        inputs = inputs.to(args.device)#, non_blocking=True)
        target = target.to(args.device)#, non_blocking=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        losses.update(loss.item(), inputs.size(0))

        if args.compute_accuracy:
            k = min(output.shape[1], 5) # If we have k less than 5 classes, show acc@k (=100%)
            acc1, acc5 = accuracy(output, target, topk=(1,k))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if args.time_meter: batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Copied to TrainingCallback in flows.py, to avoid having to add this file to the public repo
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(k,-1).float().sum()#0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args, prefix = 'Test: '):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    meters = [batch_time, losses]
    if args.compute_accuracy:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        meters += [top1, top5]
    progress = ProgressMeter(len(val_loader),meters, prefix=prefix)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            #if args.gpu is not None:
            inputs = inputs.to(args.device)#, non_blocking=True)
            target = target.to(args.device)#, non_blocking=True)

            # compute output
            output = model(inputs)
            loss = criterion(output, target)

            losses.update(loss.item(), inputs.size(0))

            # measure accuracy and record loss
            if args.compute_accuracy:
                k = min(output.shape[1], 5)
                acc1, acc5 = accuracy(output, target, topk=(1, k))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if args.compute_accuracy:
            logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        else:
            logger.info(f' * Loss {losses.avg:.3f}')

    if args.compute_accuracy:
        return losses.avg, top1.avg
    else:
        return losses.avg, None

def save_checkpoint(state, is_best, save_path='/'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


def load_checkpoint(save_path, model, optimizer, args):
    args.resume = os.path.join(save_path, 'model_best.pth.tar')
    #pdb.set_trace()
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        #if args.gpu is None:
        if args.device == 'cpu':
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = args.device#'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        if 'best_acc1' in checkpoint:
            best_acc1 = checkpoint['best_acc1']
        else:
            best_loss = checkpoint['best_loss']
        # if args.device is not 'cpu' and type(best_acc1):
        #     # best_acc1 may be from a checkpoint from a different GPU
        #     best_acc1 = best_acc1.to(args.device)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
    return model, optimizer, args



def learn_pushforward(T, X=None, Y=None, train_loader = None, valid_loader=None,
                      restarts=2, epochs=2, print_freq=10, use_ignite=True,
                      save_path = None, args=None):
    """
        min_θ'
    """
    train_args = AttrDict({'criterion': 'mse',
                           'device': args.device,
                           'optim': 'adam',
                           'lr': 1e-3,
                           'epochs': epochs,
                           'compute_accuracy': False,
                           'momentum': 0.99,
                           'save_dir': '.pshfw_models/',
                           'log_interval': 20
                           })
    train_args = default_args(train_args)

    if train_loader is None or valid_loader is None:
        dset = torch.utils.data.TensorDataset(X, Y)
        trn_size = int(0.9*len(dset))
        val_size = len(dset) - trn_size
        tst_size = 0
        ds_train, ds_valid, ds_test = torch.utils.data.random_split(dset, [trn_size, val_size, tst_size])
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size = 64, shuffle=True)
        dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = 256, shuffle=False)
    else:
        dl_train = train_loader
        dl_valid = valid_loader
    #dl_test  = torch.utils.data.DataLoader(ds_test, batch_size = 256, shuffle=False)

    best_loss = np.Inf
    pbar = tqdm(range(restarts), leave=False)
    for res in pbar:
        pbar.set_description(f'Pushmap Restart {res}')
        if use_ignite:
            T, loss = ignite_mse_train(T, dl_train, dl_valid, args=train_args)
        else:
            T, loss = train(T, dl_train, dl_valid, None, args=train_args)
        logger.info(f'Pushmap random restart #{res} loss {loss:4.2e}')

        is_best   = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            state = {
                'restart': res + 1,
                'arch': T.__class__.__name__,
                'state_dict': T.state_dict(),
                'best_loss': best_loss,
                'train_args' : train_args
            }
            dirname = os.path.dirname(save_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            torch.save(state, save_path)

        T.reset_parameters()

    logger.info("=> reloading best pushmap model")

    #if args.gpu is None:
    if args.device == 'cpu':
        checkpoint = torch.load(save_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = args.device#'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(save_path, map_location=loc)

    best_loss = checkpoint['best_loss']
    T.load_state_dict(checkpoint['state_dict'])
    logger.info(f"...best pushmap model has loss {best_loss:4.2e}")

    return T.eval(), loss

################################################################################
######################   IGNITE WRAPPER FUNCTIONS ##############################
################################################################################


def ignite_mse_train(model, dl_train, dl_valid, args=None):
    """
        Full training pipeline for : creates optimizers, criterion, trainer and evaluator

        Formerly known as ignite_train
    """
    optimizer = OPTIM[args.optim](model.parameters(), lr=args.lr)#, momentum=0.8)
    device = process_device_arg(args.device)
    criterion = nn.MSELoss().to(device)
    model = model.to(device)
    trainer = ignite.engine.create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {'mse': ignite.metrics.MeanSquaredError(), 'loss': ignite.metrics.Loss(criterion)}
    evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics,  device=device)
    to_save = {'model': model}
    def score_function(engine): # Checkpoint can only deal with 'best', so need to use -loss
        return -engine.state.metrics['loss']
    handler = ignite.handlers.Checkpoint(to_save, ignite.handlers.DiskSaver(args.save_dir, create_dir=True,require_empty=False), n_saved=1,
             filename_prefix='best', score_function=score_function, score_name="loss",
             global_step_transform=ignite.handlers.global_step_from_engine(trainer))
    evaluator.add_event_handler(ignite.engine.Events.COMPLETED, handler)
    #handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_training_loss(trainer):
        logger.info("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(dl_train)
        metrics = evaluator.state.metrics
        logger.info("Train Results - Epoch: {}  Avg loss: {:.2f}  Avg MSE: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["mse"]))
        print("Train Results - Epoch: {}  Avg loss: {:.2f} Avg MSE: {:.2f} ".format(trainer.state.epoch, metrics["loss"], metrics['mse']))
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(dl_valid)
        metrics = evaluator.state.metrics
        logger.info("Valid Results - Epoch: {}  Avg loss: {:.2f} Avg MSE: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["mse"]))
        print("Valid Results - Epoch: {}  Avg loss: {:.2f} Avg MSE: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["mse"]))

    def log_validation_results(trainer):
        evaluator.run(dl_valid)
        metrics = evaluator.state.metrics
        logger.info("Valid Results - Epoch: {}  Avg loss: {:.2f} Avg MSE: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["mse"]))
        print("Valid Results - Epoch: {}  Avg loss: {:.2f} Avg MSE: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["mse"]))

    trainer.run(dl_train, max_epochs=args.epochs)
    to_load = to_save

    checkpoint = torch.load(os.path.join(args.save_dir, handler.last_checkpoint))
    ignite.handlers.Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    evaluator.run(dl_valid)
    val_loss = evaluator.state.metrics['loss']

    return model.eval(), val_loss


def ignite_crossent_train(model, dl_train, dl_valid=None, dl_test=None, args=None):
    """
        Full training pipeline for : creates optimizers, criterion, trainer and evaluator

        Formerly known as ignite_train
    """
    optimizer = OPTIM[args.optim](model.parameters(), lr=args.lr)#, momentum=0.8)
    device = process_device_arg(args.device)
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    trainer = ignite.engine.create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {'loss': ignite.metrics.Loss(criterion), 'acc': ignite.metrics.Accuracy()}#, 'gpu': ignite.contrib.metrics.GpuInfo()}
    evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics,  device=device)
    to_save = {'model': model}

    ignite.metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')


    def score_function(engine):
        return engine.state.metrics['acc']
    handler = ignite.handlers.Checkpoint(to_save,
             ignite.handlers.DiskSaver(args.save_dir, create_dir=True,require_empty=False),
             n_saved=1, filename_prefix='best', score_function=score_function, score_name="acc",
             global_step_transform=ignite.handlers.global_step_from_engine(trainer))
    evaluator.add_event_handler(ignite.engine.Events.COMPLETED, handler)
    #handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)

    ## Uncomment only if not using progress bar below
    #@trainer.on(ignite.engine.Events.ITERATION_COMPLETED(every=args.log_interval))
    #def log_training_loss(trainer):
    #    logger.info("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(dl_train)
        metrics = evaluator.state.metrics
        logger.info("Train Results - Epoch: {}  Avg loss: {:.2f}  Avg Acc: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["acc"]))
        print("Train Results - Epoch: {}  Avg loss: {:.2f} Avg Acc: {:.2f} ".format(trainer.state.epoch, metrics["loss"], metrics['acc']))
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(dl_valid)
        metrics = evaluator.state.metrics
        logger.info("Valid Results - Epoch: {}  Avg loss: {:.2f} Avg Acc: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["acc"]))
        print("Valid Results - Epoch: {}  Avg loss: {:.2f} Avg Acc: {:.2f}".format(trainer.state.epoch,  metrics["loss"], metrics["acc"]))

    ### Info for Progress Bar
    #pdb.set_trace()
    if args.device != 'cpu':
        GpuInfo().attach(trainer, name='gpu')
    pbar = ProgressBar()
    pbar.attach(trainer,  metric_names='all')#['gpu:0 mem(%)', 'gpu:0 util(%)'])

    pbar.attach(evaluator, metric_names = ['loss', 'acc'])

    trainer.run(dl_train, max_epochs=args.epochs)
    to_load = to_save

    checkpoint = torch.load(os.path.join(args.save_dir, handler.last_checkpoint))
    ignite.handlers.Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    if dl_test is not None:
        evaluator.run(dl_test)
    else:
        evaluator.run(dl_valid)

    #pdb.set_trace()
    # shutil.copyfile(os.path.join(args.save_dir, handler.last_checkpoint),
    #                 os.path.join(args.save_dir, 'model_best.pth.tar'))

    loss = evaluator.state.metrics['loss']
    acc  = evaluator.state.metrics['acc']

    # Now save with my format
    save_checkpoint({
        'epoch': trainer.state.epoch,
        'arch': model.__class__.__name__, # TODO: How to get nlayers and add here?
        'state_dict': model.state_dict(),
        'best_loss': loss,
        'best_acc1': acc,
        'optimizer' : optimizer.state_dict(),
    }, True, args.save_dir)

    return model.eval(), acc


def ignite_crossent_eval(model, dl_test, args=None):
    """
    """
    device = process_device_arg(args.device)
    criterion = nn.CrossEntropyLoss().to(device)
    metrics = {'loss': ignite.metrics.Loss(criterion), 'acc': ignite.metrics.Accuracy()}
    model = model.to(device)
    evaluator = ignite.engine.create_supervised_evaluator(model, metrics=metrics,  device=device)
    # def score_function(engine):
    #     return engine.state.metrics['acc']
    # handler = ignite.handlers.Checkpoint(to_save,
    #          ignite.handlers.DiskSaver(args.save_dir, create_dir=True,require_empty=False),
    #          n_saved=1, filename_prefix='best', score_function=score_function, score_name="acc",
    #          global_step_transform=ignite.handlers.global_step_from_engine(trainer))
    # evaluator.add_event_handler(ignite.engine.Events.COMPLETED, handler)
    #handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)

    #checkpoint = torch.load(os.path.join(args.save_dir, handler.last_checkpoint))
    #ignite.handlers.Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
    evaluator.run(dl_test)
    loss = evaluator.state.metrics['loss']
    acc  = evaluator.state.metrics['acc']

    logger.info(f' * Loss {loss:.3f}  Acc@1 {100*acc:.3f}')

    return loss, acc



########################################################################################





# def process_function(engine, batch):
#     model.train()
#     optimizer.zero_grad()
#     x, y = batch
#     x = x.to(device)
#     y = y.to(device)
#     x = x.view(-1, 784)
#     x_pred, mu, logvar = model(x)
#     BCE = bce_loss(x_pred, x)
#     KLD = kld_loss(x_pred, x, mu, logvar)
#     loss = BCE + KLD
#     loss.backward()
#     optimizer.step()
#     return loss.item(), BCE.item(), KLD.item()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
# criterion = nn.NLLLoss()
#
# trainer = create_supervised_trainer(model, optimizer, criterion)
#
# val_metrics = {
#     "accuracy": Accuracy(),
#     "nll": Loss(criterion)
# }
# evaluator = create_supervised_evaluator(model, metrics=val_metrics)
#
# @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
# def log_training_loss(trainer):
#     print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
#
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(trainer):
#     evaluator.run(train_loader)
#     metrics = evaluator.state.metrics
#     print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
#           .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
#
# @trainer.on(Events.EPOCH_COMPLETED)
# def log_validation_results(trainer):
#     evaluator.run(val_loader)
#     metrics = evaluator.state.metrics
#     print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
#           .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))
#
# trainer.run(train_loader, max_epochs=100)
