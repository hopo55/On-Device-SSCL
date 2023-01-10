import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import learners
import dataloaders
from dataloaders.utils import *


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # prepare dataloader
    Dataset = None
    if args.dataset == 'CIFAR10':
        Dataset = dataloaders.iCIFAR10
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
        num_classes = 100
    elif args.dataset == 'TinyIMNET':
        Dataset = dataloaders.iTinyIMNET
        num_classes = 200

    # load tasks
    class_order = np.arange(num_classes).tolist()
    class_order_logits = np.arange(num_classes).tolist()
    tasks = []
    tasks_logits = []
    p = 0
    while p < num_classes:
        inc = args.other_split_size if p > 0 else args.first_split_size
        tasks.append(class_order[p:p+inc])
        tasks_logits.append(class_order_logits[p:p+inc])
        p += inc
    num_tasks = len(tasks)
    task_names = [str(i+1) for i in range(num_tasks)]

    # number of transforms per image
    # Use fix-match loss with classifier
    k = 1
    if args.fm_loss: 
        k = 2 # Append transform image and buffer image
    ky = 1 # Not append transform for memory buffer
    
    # datasets and dataloaders
    train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug)
    train_transformb = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, hard_aug=True)
    test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug)

    print('Labeled train dataset')
    train_dataset = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = True,
                            download=True, transform=TransformK(train_transform, train_transform, ky), l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=args.seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    print('Unlabeled train dataset')
    train_dataset_ul = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = False,
                            download=True, transform=TransformK(train_transform, train_transformb, k), l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=args.seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    print('Test dataset')
    test_dataset  = Dataset(args.dataroot, args.dataset, train=False,
                            download=False, transform=test_transform, l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=args.seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    
    # in case tasks reset
    tasks = train_dataset.tasks

    # Prepare the Learner (model)
    learner_config = {'num_classes': num_classes,
                      'lr': args.lr,
                      'ul_batch_size': args.ul_batch_size,
                      'momentum': args.momentum,
                      'weight_decay': args.weight_decay,
                      'schedule': args.schedule,
                      'schedule_type': args.schedule_type,
                      'model_type': args.model_type,
                      'model_name': args.model_name,
                      'out_dim': args.out_dim,
                      'optimizer': args.optimizer,
                      'gpuid': args.gpuid,
                      'pl_flag': args.pl_flag, # use pseudo-labeled ul data for DM
                      'fm_loss': args.fm_loss, # Use fix-match loss with classifier -> Consistency Regularization / eq.4 -> unsupervised loss
                      'weight_aux': args.weight_aux,
                      'memory': args.memory,
                      'distill_loss': args.distill_loss,
                      'FT': args.FT, # finetune distillation -> 이거 필요한가???
                      'DW': args.DW, # dataset balancing
                      'num_labeled_samples': args.labeled_samples,
                      'num_unlabeled_samples': args.unlabeled_task_samples,
                      'super_flag': args.l_dist,
                      'no_unlabeled_data': args.no_unlabeled_data,
                      'last': None
                      }

    learner = learners.sscl.SSCL(learner_config)

    acc_table = OrderedDict()
    acc_table_pt = OrderedDict()
    run_ood = {}

    log_dir = "outputs/CIFAR100-10k/realistic/dm"
    save_table = []
    save_table_pc = -1 * np.ones((num_tasks,num_tasks))
    pl_table = [[],[],[],[]]
    temp_dir = log_dir + '/temp'
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    # Training
    max_task = -1
    if max_task > 0:
        max_task = min(max_task, len(task_names))
    else:
        max_task = len(task_names)

    for i in range(max_task):
        # set seeds
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        train_name = task_names[i]
        print('======================', train_name, '=======================')

        # load dataset for task
        task = tasks_logits[i]
        # prev where classes learned so far are stored
        # ex) frist task = [], second task = [0, 1, 2, 3, 4]
        prev = sorted(set([k for task in tasks_logits[:i] for k in task])) 

        # current class와 prev class all load
        train_dataset.load_dataset(prev, i, train=True)
        train_dataset_ul.load_dataset(prev, i, train=True)
        out_dim_add = len(task)

        # load dataset with memory(coreset)
        train_dataset.append_coreset(only=False)

        # load dataloader
        train_loader_l = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader_ul = DataLoader(train_dataset_ul, batch_size=args.ul_batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader_ul_task = DataLoader(train_dataset_ul, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=int(args.workers / 2))
        train_loader = dataloaders.SSLDataLoader(train_loader_l, train_loader_ul) # return labeled data, unlabeled data

        # add valid class to classifier
        learner.add_valid_output_dim(out_dim_add) # return number of classes learned to the current task

        # Learn
        # load test dataset dataloader
        test_dataset.load_dataset(prev, i, train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

        model_save_dir = log_dir + '/models/repeat-'+str(seed+1)+'/task-'+task_names[i]+'/'
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        learner.learn_batch(train_loader, train_dataset, train_dataset_ul, model_save_dir, test_loader)
        
        # Evaluate
        acc_table[train_name] = OrderedDict()
        acc_table_pt[train_name] = OrderedDict()
        for j in range(i+1):
            val_name = task_names[j]
            print('validation split name:', val_name)
            test_dataset.load_dataset(prev, j, train=True)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

            # validation
            acc_table[val_name][train_name] = learner.validation(test_loader)
            save_table_pc[i,j] = acc_table[val_name][train_name]

            # past task validation
            acc_table_pt[val_name][train_name] = learner.validation(test_loader, task_in = tasks_logits[j])

        save_table.append(np.mean([acc_table[task_names[j]][train_name] for j in range(i+1)]))

        # Evaluate PL
        if i+1 < len(task_names):
            test_dataset.load_dataset(prev, len(task_names)-1, train=False)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
            stats = learner.validation_pl(test_loader)
            names = ['stats-fpr','stats-tpr','stats-de']
            for ii in range(3):
                pl_table[ii].append(stats[ii])
                save_file = temp_dir + '/'+names[ii]+'_table.csv'
                np.savetxt(save_file, np.asarray(pl_table[ii]), delimiter=",", fmt='%.2f')

            run_ood['tpr'] = pl_table[1]
            run_ood['fpr'] = pl_table[0]
            run_ood['de'] = pl_table[2]

        # save temporary results
        save_file = temp_dir + '/acc_table.csv'
        np.savetxt(save_file, np.asarray(save_table), delimiter=",", fmt='%.2f')
        save_file_pc = temp_dir + '/acc_table_pc.csv'
        np.savetxt(save_file_pc, np.asarray(save_table_pc), delimiter=",", fmt='%.2f')

    return acc_table, acc_table_pt, task_names, run_ood

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--seed', default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="CIFAR10|CIFAR100|TinyIMNET")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate on fold of training dataset rather than testing data')
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--workers', type=int, default=8, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ul_batch_size', type=int, default=128)
    # SSL Args
    parser.add_argument('--fm_loss', default=True, action='store_true', help='Use fix-match loss with classifier (WARNING: currently only pseudolabel)')
    parser.add_argument('--labeled_samples', type=int, default=10000, help='Number of labeled samples in ssl')
    parser.add_argument('--unlabeled_task_samples', type=int, default=-1, help='Number of unlabeled samples in each task in ssl')
    # CL Args
    parser.add_argument('--first_split_size', type=int, default=5, help="size of first CL task")
    parser.add_argument('--other_split_size', type=int, default=5, help="size of remaining CL tasks")
    parser.add_argument('--train_aug', dest='train_aug', default=True, action='store_true', help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=True, action='store_true', help="Randomize the classes in splits")
    parser.add_argument('--l_dist', type=str, default='super', help="vanilla|super")
    parser.add_argument('--ul_dist', type=str, default=None, help="none|vanilla|super - if none, copy l dist")
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--schedule_type', type=str, default='cosine',
                        help="decay, cosine")

    parser.add_argument('--FT', default=False, action='store_true', help='finetune distillation')

    # OOD Args
    parser.add_argument('--ood_model_name', type=str, default=None, help="The name of actual model for the backbone ood")
    parser.add_argument('--tpr', type=float, default=0.95, help="tpr for ood calibration of class network")
    parser.add_argument('--oodtpr', type=float, default=0.95, help="tpr for ood calibration of ood network")

    # SSL Args
    parser.add_argument('--weight_aux', type=float, default=1.0, help="Auxillery weight, usually used for trading off unsupervised and supervised losses")
    parser.add_argument('--pl_flag', default=False, action='store_true', help='use pseudo-labeled ul data for DM')
    
    # GD Args
    parser.add_argument('--no_unlabeled_data', default=False, action='store_true')
    parser.add_argument('--distill_loss', nargs="+", type=str, default='C', help='P, C, Q')
    parser.add_argument('--co', type=float, default=1., metavar='R',
                    help='out-of-distribution confidence loss ratio (default: 0.)')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--max_task', type=int, default=-1, help="number tasks to perform; if -1, then all tasks")
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True

    acc_table, acc_table_pt, task_names, run_ood = run(args)
    print(acc_table)