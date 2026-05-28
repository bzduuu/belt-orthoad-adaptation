import os
import sys
import argparse
import logging
import time
import copy
import random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as vision

from metric import evaluate, checkout_objective
from optimizer import *
from dataset import *
from model import *
from utils import *
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dataroot', default=os.environ.get('DATA', '.'),
                        help='Path to the dataset')
    parser.add_argument('--dataset', default=MVTecAD.name,
                        choices=['mvtecad', 'kolektor', 'kolektor2', 'mstc'],
                        help='Dataset to train')
    parser.add_argument('--batch-size', type=int,
                        default=2, help='Input batch size')
    parser.add_argument('--drop-last', type=bool,
                        default=False, help='Drop last for training split')
    parser.add_argument('--fold', type=int,
                        default=0, help='Cross validation fold for KolektorSDD')
    parser.add_argument('--scale', type=str,
                        default='half', help='Input image scale for KolektorSDD')
    parser.add_argument('--category', default='carpet',
                        help='Dataset category to train')
    parser.add_argument('--workers', type=int, default=0,
                        help='The number of workers for data loaders')
    parser.add_argument('--model', default='wide_resnet50_2', type=str,
                        choices=['resnet18', 'wide_resnet50_2', 
                                 'mobilenetv3_large', 'mobilenetv3_small'],
                        help='Anomaly detection model')
    parser.add_argument('--method', default='mah', type=str,
                        help='Training loss to optimize')
    parser.add_argument('--approx', default='ortho', type=str,
                        choices=['ortho', 'sample', 'gaussian', 
                                 'global', 'lowrank', 'lowranki', 'null'],
                        help='Mahalanobis distance approximation method')
    parser.add_argument('--k', type=int, default=300, help='k-rank')
    parser.add_argument('--metric', type=str, default='auproc',
                        help='Evaluation metric', 
                        choices=['auproc', 'auroc', 'fpr', 'ap','none'])
    parser.add_argument('--fpr', type=float, default=.3,
                        help='The false positive rate cut for PRO-curve')
    parser.add_argument('--recall', default=.95, type=float,
                        help='Normalize the score using a validation split')
    parser.add_argument('--nSamples', type=int, default=1000,
                        help='The number of samples for PRO-curve')
    parser.add_argument('--benchmark-inference', action='store_true',
                    default=False, help='Measure test inference speed')
    parser.add_argument('--benchmark-warmup', type=int, default=10,
                    help='Number of warmup test batches ignored in benchmark')
    parser.add_argument('--use-val-norm', action='store_true',
                    default=False,
                    help='Normalize anomaly scores using validation '
                         'set statistics (per-position mean/std on good '
                         'samples). Helps to compensate for textured '
                         'background of belt surface.')

    parser.add_argument('--verbose', action='store_true',
                        default=False, help='Log verbosity')  # for analysis
    parser.add_argument('--experiment', default=None,
                        help='Where to store models')
    parser.add_argument('--report', default='results.out', help='Report path')
    parser.add_argument('--startidx', type=int, default=0,
                        help='Starting index for the test split of mSTC')
    parser.add_argument('--label', default='', help='Experimental label')
    parser.add_argument('--seed', default='1111', help='Random seed')

    # default settings
    args = parser.parse_args()

    # override default category
    if STCAD.name == args.dataset:
        args.category = STCAD.category
    elif KolektorSDD.name == args.dataset:
        args.category = KolektorSDD.category
    elif KolektorSDD2.name == args.dataset:
        args.category = KolektorSDD2.category        

    # experiment preparation
    mkdirs(['logs', 'cache'])
    if args.experiment is None:
        args.experiment = get_default_experiment(args)
    mkdirs([args.experiment])
    set_logging_config(args.experiment)

    # logging
    logger = logging.getLogger(get_basename_without_ext(__file__))
    logger.info(greetings())
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    # reproducibility
    if STCAD.name == args.dataset:
        random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # feature extractor
    if 'resnet' in args.model:
        model = getattr(vision.models, args.model)(pretrained=True)
        model = SpadeResNet(model, label=args.label)
    elif 'mobilenetv3' in args.model:
        from mobilenetv3 import mobilenetv3_large, mobilenetv3_small
        if 'small' in args.model:
            model = mobilenetv3_small()
            model.load_state_dict(torch.load(
                '../mobilenetv3.pytorch/pretrained/mobilenetv3-small-55df8e1f.pth'))
        else:
            model = mobilenetv3_large()
            model.load_state_dict(torch.load(
                '../mobilenetv3.pytorch/pretrained/mobilenetv3-large-1cd25616.pth'))
        model = SpadeMobilenetV3(model, label=args.label)
    else:
        raise NotImplementedError()
    logger.info('Model: {}, nParams: {}'.format(args.model, num_params(model)))
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    # data, optimizer preparation
    loaders = checkout_dataloader(args, ['train', 'val', 'test'])  # train, val
    logger.info('Hey dude, for train, nSamples={:4d}, nIters={:3d}'.format(
        *repr_loader(loaders[0])))
    logger.info('          for valid, nSamples={:4d}, nIters={:3d}'.format(
        *repr_loader(loaders[1])))
    logger.info('          for test, nSamples={:4d}, nIters={:3d}'.format(
        *repr_loader(loaders[2])))

    # features
    logger.info('Extract features...')
    X = None
    N = len(loaders[0].dataset)
    B = args.batch_size
    for i, data in enumerate(loaders[0]):
        x, y, a = data[:3]
        x = x.to(device)
        out = model(x)

        if X is None:
            b, c, h, w = out.size()
            X = torch.Tensor(h, w, args.k, args.k).zero_().to(device)  # covariance
            X_mean = torch.Tensor(h, w, args.k).zero_().to(device)  # mean
            W = MahEvaluator.get_embedding(c, args.k, args.approx).to(device)
            print('Covariance size: {}'.format(X.shape))

        out = torch.einsum('bchw, cd -> bdhw', out, W)
        X += torch.einsum('bchw, bdhw -> hwcd', (out, out))
        X_mean += out.sum(0).transpose(0, 1).transpose(1, 2)
    
    X /= N
    X_mean /= N
    X -= torch.einsum('hwc, hwd -> hwcd', (X_mean, X_mean))  # unbiased
    X = X
    X_mean = X_mean
    
    # to reproduce the PaDiM results (Defard et al., 2021)
    EPSILON = 1e-2 if STCAD.name != args.dataset else 3e-1

    # evaluator 
    model.evaluator = MahEvaluator(X, X_mean, W, args.k, args.approx, 
                                   num_samples=N, eps=EPSILON)

    # objective
    objective = checkout_objective(args)

    if not args.use_val_norm:  # default: do not use validation scores
        val_scores = [torch.zeros(1,1), torch.ones(1,1)]
        logger.info('Validation score normalization: DISABLED')
    else:  # validation score normalization
        logger.info('Validation score normalization: ENABLED')
        logger.info('Computing the means and stds for a validation set')

        val_scores = [[], []]
        reduction = True

        for i, data in enumerate(tqdm(loaders[1])):
            x, y, a = data[:3]
            x = x.to(device)
            B = x.size(0)
            means, stds = objective(x, model, args=args, reduction=reduction)
            # means, stds = mse_per_stage_score(x, teacher, student, args)
            val_scores[0].append(means.cpu())
            val_scores[1].append(stds.cpu())

        val_scores[0] = torch.cat(val_scores[0], dim=0)
        val_scores[1] = torch.cat(val_scores[1], dim=0)

        if reduction:
            val_scores = [flatten(val_scores[0], dim=1).mean(-1), 
                          flatten(val_scores[1], dim=1).pow(2).mean(-1).pow(.5)]
            val_scores = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) \
                          for x in val_scores]
        else:
            val_scores = [val_scores[0].mean(0, keepdim=True),
                          val_scores[1].pow(2).mean(0, keepdim=True).pow(.5)]

        logger.info('val_scores mean: {}'.format(
            flatten(val_scores[0], 1).squeeze(-1)[:20].numpy()))
        logger.info('val_scores std: {}'.format(
            flatten(val_scores[1], 1).squeeze(-1)[:20].numpy()))

    # calculate scores
    scores = 0
    loader = loaders[-1]

    pred = None
    gt = None

    benchmark_batch_times = []
    benchmark_image_times = []

    for j, data in enumerate(tqdm(loader)):
        x, y, a, c = data[:4]

        if pred is None:
            mask = torch.Tensor(len(loader.dataset),
                                x.size(2), x.size(3)).zero_()
            pred = torch.Tensor(len(loader.dataset),
                                x.size(2), x.size(3)).zero_()
            gt = torch.Tensor(len(loader.dataset),
                              x.size(2), x.size(3)).zero_()
            typ = torch.LongTensor(len(loader.dataset))

        gt[j * loader.batch_size: j * loader.batch_size + x.size(0)] = a
        mask[j * loader.batch_size: j * loader.batch_size + x.size(0)] = a
        typ[j * loader.batch_size: j * loader.batch_size + x.size(0)] = y

        current_batch_size = x.size(0)

        if args.benchmark_inference and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        x = x.to(device)
        score = objective(x, model, args, val_scores)
        score = score.cpu()

        if args.benchmark_inference and torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        if args.benchmark_inference and j >= args.benchmark_warmup:
            elapsed = t1 - t0
            benchmark_batch_times.append(elapsed)
            benchmark_image_times.append(elapsed / current_batch_size)

        pred[j * loader.batch_size: j *
                 loader.batch_size + x.size(0)] = score

    if args.benchmark_inference:
        if len(benchmark_batch_times) > 0:
            batch_times = np.array(benchmark_batch_times)
            image_times = np.array(benchmark_image_times)

            mean_batch_ms = batch_times.mean() * 1000
            median_batch_ms = np.median(batch_times) * 1000
            p95_batch_ms = np.percentile(batch_times, 95) * 1000
            p99_batch_ms = np.percentile(batch_times, 99) * 1000

            mean_image_ms = image_times.mean() * 1000
            median_image_ms = np.median(image_times) * 1000
            p95_image_ms = np.percentile(image_times, 95) * 1000
            p99_image_ms = np.percentile(image_times, 99) * 1000

            fps_images = 1.0 / image_times.mean()
            fps_batches = 1.0 / batch_times.mean()

            benchmark_text = (
                "\n"
                "Inference benchmark\n"
                f"model: {args.model}\n"
                f"k: {args.k}\n"
                f"batch_size: {args.batch_size}\n"
                f"workers: {args.workers}\n"
                f"device: {device}\n"
                f"warmup_batches: {args.benchmark_warmup}\n"
                f"measured_batches: {len(batch_times)}\n"
                f"mean_batch_ms: {mean_batch_ms:.3f}\n"
                f"median_batch_ms: {median_batch_ms:.3f}\n"
                f"p95_batch_ms: {p95_batch_ms:.3f}\n"
                f"p99_batch_ms: {p99_batch_ms:.3f}\n"
                f"mean_image_ms: {mean_image_ms:.3f}\n"
                f"median_image_ms: {median_image_ms:.3f}\n"
                f"p95_image_ms: {p95_image_ms:.3f}\n"
                f"p99_image_ms: {p99_image_ms:.3f}\n"
                f"fps_images: {fps_images:.2f}\n"
                f"fps_batches: {fps_batches:.2f}\n"
            )

            print(benchmark_text)
            logger.info(benchmark_text)

            with open(os.path.join(args.experiment, "benchmark_inference.txt"),
                      "w", encoding="utf-8") as f:
                f.write(benchmark_text)
        else:
            logger.info("Inference benchmark skipped: not enough batches after warmup.")

    n = num_samples_per_category_to_save = len(loader.dataset)
    types = []
    product_ids = []
    # random sampling for visualization
    # m = torch.randperm(len(loader.dataset))[:n]
    m = torch.LongTensor(list(range(n)))
    for j in range(n):
        types.append(loader.dataset.labels[typ[m[j]]])
        if 'kolektor' == args.dataset:
            product_ids.append(loader.dataset.product_ids[j])
        else:
            product_ids.append(m[j].data)
    # call clone() to save disk space
    data = (mask[m].clone(), pred[m].clone(), types, product_ids)
    torch.save(data, os.path.join(
        args.experiment, '{}.pth'.format(loader.dataset.category)))
    if args.metric == 'none':
        logger.info('Metric skipped: test set contains only normal belt images.')
        return
    pred = pred.to(device)
    gt = gt.to(device)
    typ = typ.to(device)
    scores = evaluate(pred, gt, method=args.metric, at_fpr=args.fpr,
                         num_samples=args.nSamples, verbose=True)[0]
    logger.info('{:10} {:1.4f}'.format('SEG ' + args.metric.upper(), scores))
    with open(args.report, 'a') as f:
        f.write('{} {:10} {:.4f}\n'.format(args.metric, loader.dataset.category, scores))

    if args.metric in ['auroc']:
        det_pred = pred.max(1, keepdim=True)[0].max(2, keepdim=True)[0]
        det_roc, recall_to_precision = evaluate(
            det_pred, (typ != 0).unsqueeze(1).unsqueeze(2),
            method=args.metric, at_fpr=1.,
            num_samples=pred.size(0), verbose=True)
        logger.info('{:10} {:1.4f}'.format('DET ' + args.metric.upper(), 
            det_roc))
        with open(args.report + '.det', 'a') as f:
            f.write('{:10} {:.4f}\n'.format(loader.dataset.category, det_roc))

if __name__ == '__main__':
    main()
