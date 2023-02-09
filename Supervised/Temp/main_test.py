# import os
# import time
# import random
# import argparse
# import numpy as np

# import dataloader
# import data_generator
# from models import SLDA, NCM, StreamSoftmax
# from Temp.utils import *

# import torch
# import torch.backends.cudnn as cudnn


# def compute_accuracies(loader, classifier):
#     probas, y_test_init = classifier.evaluate_(loader)
#     top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
#     return probas, top1, top5

# def update_accuracies(class_remap, curr_max_class, classifier, accuracies, save_dir, batch_size, shuffle, dataset):
#     seen_classes_test_loader = get_class_data_loader(args, class_remap, False, 0, curr_max_class, batch_size=batch_size,
#                                                      shuffle=shuffle, dataset=dataset, return_item_ix=True)
#     seen_probas, seen_top1, seen_top5 = compute_accuracies(seen_classes_test_loader, classifier)

#     print('\nSeen Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%' % (0, curr_max_class - 1, seen_top1, seen_top5))
#     accuracies['seen_classes_top1'].append(float(seen_top1))
#     accuracies['seen_classes_top5'].append(float(seen_top5))

#     # save accuracies and predictions out
#     save_accuracies(accuracies, min_class_trained=0, max_class_trained=curr_max_class, save_path=save_dir)
#     save_predictions(seen_probas, 0, curr_max_class, save_dir)

# def streaming_class_iid_training(args, classifier, class_remap):
#     start_time = time.time()
#     # start list of accuracies
#     accuracies = {'seen_classes_top1': [], 'seen_classes_top5': []}
#     # save_name = "model_weights_min_trained_0_max_trained_%d"

#     # loop over all data and compute accuracy after every "batch"
#     for curr_class_ix in range(0, args.num_classes, args.class_increment):
#         max_class = min(curr_class_ix + args.class_increment, args.num_classes)
#         print(curr_class_ix)

#         # get training loader for current batch
#         data_loader = dataloader.dataloader(args)

#         # fit model
#         classifier.train_(data_loader, curr_class_ix)

#         if curr_class_ix != 0 and ((curr_class_ix + 1) % args.evaluate_increment == 0):
#             # print("\nEvaluating classes from {} to {}".format(0, max_class))
#             # output accuracies to console and save out to json file
#             update_accuracies(class_remap, max_class, classifier, accuracies, args.save_dir, args.batch_size, shuffle=False, dataset=args.dataset)
#             # classifier.save_model(save_dir, save_name % max_class)

#     # print final accuracies and time
#     probas, y_test = classifier.evaluate_(data_loader, curr_class_ix)
#     top1, top5 = accuracy(probas, y_test, topk=(1, 5))
#     accuracies['seen_classes_top1'].append(float(top1))
#     accuracies['seen_classes_top5'].append(float(top5))

#     # save accuracies, predictions, and model out
#     save_accuracies(accuracies, min_class_trained=0, max_class_trained=args.num_classes, save_path=args.save_dir)
#     save_predictions(probas, 0, args.num_classes, args.save_dir)
#     classifier.save_model(args.save_dir, "model_weights_final")

#     end_time = time.time()
#     print('\nModel Updates: ', classifier.num_updates)
#     print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
#     print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))


# def streaming_iid_training(args, classifier):
#     start_time = time.time()
#     # start list of accuracies
#     accuracies = {'top1': [], 'top5': []}
#     # save_name = "model_weights_%d"

#     train_loader = get_iid_data_loader(args, True, batch_size=args.batch_size, shuffle=True, dataset=args.dataset, return_item_ix=True)
#     test_loader = get_iid_data_loader(args, False, batch_size=args.batch_size, shuffle=False, dataset=args.dataset, return_item_ix=True)

#     start = 0
#     for batch_x, batch_y, batch_ix in train_loader:
#         # fit model
#         classifier.fit_batch(batch_x, batch_y, batch_ix)

#         end = start + len(batch_x)
#         # TODO: decide if we want to compute performance between batches
#         start = end

#     # print final accuracies and time
#     probas, y_test = classifier.evaluate_(test_loader)
#     top1, top5 = accuracy(probas, y_test, topk=(1, 5))
#     accuracies['top1'].append(float(top1))
#     accuracies['top5'].append(float(top5))

#     # save accuracies, predictions, and model out
#     save_accuracies(accuracies, min_class_trained=0, max_class_trained=args.num_classes, save_path=args.save_dir)
#     save_predictions(probas, 0, args.num_classes, args.save_dir)
#     classifier.save_model(args.save_dir, "model_weights_final")

#     end_time = time.time()
#     print('\nModel Updates: ', classifier.num_updates)
#     print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
#     print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))

# def evaluate(args, classifier):
#     start_time = time.time()
#     # start list of accuracies
#     accuracies = {'top1': [], 'top5': []}

#     test_loader = get_iid_data_loader(args, False, batch_size=args.batch_size, shuffle=False, dataset=args.dataset, return_item_ix=True)

#     # print final accuracies and time
#     probas, y_test = classifier.evaluate_(test_loader)
#     top1, top5 = accuracy(probas, y_test, topk=(1, 5))
#     accuracies['top1'].append(float(top1))
#     accuracies['top5'].append(float(top5))

#     end_time = time.time()
#     print('\nModel Updates: ', classifier.num_updates)
#     print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
#     print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # General Settings
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--root', type=str, default='./data/')
#     parser.add_argument('--ckpt', type=str, default=None)
#     parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100'])
#     parser.add_argument('--image_size', type=int, default=32)
#     parser.add_argument('--batch_size', type=int, default=512)  # batch size for testing
#     parser.add_argument('--test_size', type=int, default=64)
#     parser.add_argument('--num_workers', type=int, default=8)

#     # Model Settings
#     parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large'])
#     parser.add_argument('--cl_model', type=str, default='ncm', choices=['slda', 'fine_tune', 'replay', 'ncm'])
#     parser.add_argument('--evaluate', type=bool_flag, default=False)
#     parser.add_argument('--data_ordering', default='class_iid', choices=['class_iid', 'iid'])
#     parser.add_argument('--num_classes', type=int, default=10)  # total number of classes in the dataset
#     parser.add_argument('--class_increment', type=int, default=1)
#     parser.add_argument('--lr', type=float, default=0.1) # 0.0001
#     parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
#     parser.add_argument('--buffer_size', type=int, default=730)
#     parser.add_argument('--replay_size', type=int, default=50)

#     # SLDA parameters
#     parser.add_argument('--shrinkage', type=float, default=1e-4)  # shrinkage for SLDA/Naive Bayes
#     parser.add_argument('--streaming_update_sigma', type=bool_flag, default=True)  # true to update covariance online

#     args = parser.parse_args()

#     args.input_feature_size = get_feature_size(args.arch)

#     ## GPU Setup
#     torch.cuda.set_device(args.device)
#     # Set Seed
#     cudnn.deterministic = True
#     cudnn.benchmark = False
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     # Dataset Generator
#     if 'CIFAR' in args.dataset:
#         data_generator.__dict__['CIFAR_Generator'](args)
#     root = os.path.join(args.root, args.dataset)

#     backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

#     # setup continual model
#     print('\nUsing the %s continual learning model.' % args.cl_model)
#     if args.cl_model == 'slda':
#         classifier = SLDA.StreamingLDA(args.input_feature_size, args.num_classes,
#                                   shrinkage_param=args.shrinkage, streaming_update_sigma=args.streaming_update_sigma)
#     elif args.cl_model == 'fine_tune':
#         classifier = StreamSoftmax.StreamingSoftmax(args.input_feature_size, args.num_classes, use_replay=False, lr=args.lr, weight_decay=args.wd)
#         if args.ckpt is not None:
#             classifier.load_model(args.ckpt)
#     elif args.cl_model == 'replay': # using this for replay
#         classifier = StreamSoftmax.StreamingSoftmax(args.input_feature_size, args.num_classes, use_replay=True, lr=args.lr, weight_decay=args.wd, replay_samples=args.replay_size, max_buffer_size=args.buffer_size)
#         if args.ckpt is not None:
#             classifier.load_model(args.ckpt)
#     elif args.cl_model == 'ncm':
#         classifier = NCM.NearestClassMean(args.input_feature_size, args.num_classes, backbone=backbone)
#         if args.ckpt is not None:
#             classifier.load_model(args.ckpt)
#     else:
#         raise NotImplementedError

#     if args.evaluate:
#         print("evaluate")
#         evaluate(args, classifier)
#     else:
#         if args.data_ordering == 'class_iid':
#             # get class ordering
#             class_remap = remap_classes(args.num_classes, args.seed) # shuffle class idx
#             streaming_class_iid_training(args, classifier, class_remap)
#         elif args.data_ordering == 'iid':
#             streaming_iid_training(args, classifier)
#         else:
#             raise NotImplementedError