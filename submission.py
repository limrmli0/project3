import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision.transforms as T
import einops

from datasets import Proj3_Dataset
from models import *

np.random.seed(240604)


def get_args_parser():
    # Usage example: python submission.py --arch_ver ver1 --ckpt_path outputs/{path_to_your_model}/ckpt/ep30.pt --num_crop {1/3/5/10}
    # You can add more arguments if you want
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--arch_ver', default='ver1')
    argparser.add_argument('--num_crop', default=1, type=int, choices=[1, 3, 5, 10])
    argparser.add_argument('--ckpt_path', required=True)
    args = argparser.parse_args()

    return args

def run_eval(net, data_loader):
    net.eval()

    output_logit = []
    with torch.no_grad():
        for idx, img in enumerate(data_loader):
            num_batch = img.shape[0]
            if multi_crop_flag:
                img = einops.rearrange(img, 'b v c h w -> (b v) c h w', b=num_batch)

            img = img.to(device)
            img = (img - img_mean) / img_std
            out = net(img)
            prob = out.softmax(dim=1)
            if multi_crop_flag:
                prob = einops.rearrange(prob, '(b v) nc -> b v nc', b=num_batch).mean(dim=1)

            output_logit.append(prob.cpu().numpy())

    output_logit = np.concatenate(output_logit)
    output_cls = np.argmax(output_logit, axis=1)

    return output_cls


if __name__ == '__main__':
    args = get_args_parser()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 2
    batch_size = 8
    num_cls = 50
    arch_ver = args.arch_ver
    model_choices = {'ver1': R34_ver1}
    crop_choices = {1: T.CenterCrop, 5: T.FiveCrop, 10: T.TenCrop}
    img_size = 256
    crop_size = 224
    multi_crop_flag = True if args.num_crop > 1 else False

    # dataset loading
    test_subm = pd.read_csv('datasets/test_subm.csv')
    val_transform = T.Compose([T.Resize(img_size),
                               crop_choices[args.num_crop](crop_size),
                               T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])) \
                                   if multi_crop_flag else T.ToTensor(),
                               ])

    test_subm_dataset = Proj3_Dataset(test_subm, 'test', val_transform)
    test_subm_loader = DataLoader(test_subm_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    print("Test dataset: #", len(test_subm_dataset))

    # load the trained model
    net = model_choices[arch_ver](num_cls=num_cls).to(device)
    net.load_state_dict(torch.load(args.ckpt_path))
    summary(net, input_size=(3, crop_size, crop_size)) # print model summary (# of parameters, etc.)

    # Prediction
    img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(device)
    img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(device)
    output_cls = run_eval(net, test_subm_loader)

    # Save the submission file
    SID = 1234  # Change this to your student ID
    test_subm['cls'] = output_cls
    test_subm.to_csv(f'datasets/{SID}_test_subm.csv', index=False)