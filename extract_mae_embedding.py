# @Time    : 6/12/23 11:18 AM
# @Author  : bbbdbbb
# @File    : extract_mae_embedding.py
# @Description : extract embedding from pretrain models of mae

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_

import sys

sys.path.append('../../')
#import config
from dataset import FaceDataset

from mae import models_vit
from mae.util.pos_embed import interpolate_pos_embed
from scipy.stats import norm
import scipy.stats as stats


def extract(data_loader, model):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in data_loader:
            images = images.to(device)
            outputs = model(images)
            embedding = outputs
            print("embedding shape: ", embedding.shape)

            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='', help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
 
    # parser.add_argument('--pretrain_model', type=str, default='mae_random_300', help='pth of pretrain MAE model')
    # parser.add_argument('--pretrain_model', type=str, default='mae_DFEW_ck16', help='pth of pretrain MAE model')
    # parser.add_argument('--feature_name', type=str, default='mae_DFEW_ck16', help='pth of pretrain MAE model')
    parser.add_argument('--pretrain_model', type=str, default='mae_checkpoint-340', help='pth of pretrain MAE model')
    parser.add_argument('--feature_name', type=str, default='mae_340', help='pth of pretrain MAE model')

    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train vit_large_patch16 vit_huge_patch14')
    parser.add_argument('--nb_classes', default=7, type=int, help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=512, type=int)

    params = parser.parse_args()

    print(f'==> Extracting mae embedding...')
    # face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    # save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'{params.feature_name}_{params.feature_level[:3]}')
    # if not os.path.exists(save_dir): os.makedirs(save_dir)
    if params.dataset == "MER2023":
        face_dir = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/dataset-process/pyfeat_face/crop_images"
        save_dir = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/features/AIHUB_mae_340_UTT"
    elif params.dataset == "EMER":
        face_dir = "/home/amax/big_space/datasets/MER2024/EMER/all_face"
        save_dir = "/home/amax/big_space/datasets/MER2024/EMER/features_tmp/mae_340_UTT"
    elif params.dataset == "MER2024":
        face_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/all_face"
        save_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/features_tmp/mae_340_UTT"
    elif params.dataset == "MER2024_20000":
        face_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/all_face"
        save_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/features_20000_tmp/mae_340_UTT"
    elif params.dataset == "DFEW":
        face_dir = "/home/amax/big_space/datasets/DFEW/dataset-process/openface_face"
        save_dir = "/home/amax/big_space/datasets/DFEW/dataset-process/features_tmp/mae_340_UTT"

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model
    model = models_vit.__dict__[params.model](
        num_classes=params.nb_classes,
        drop_path_rate=params.drop_path,
        global_pool=params.global_pool,
    )
    if True:
        # checkpoint_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'mae', params.pretrain_model + '.pth')
        checkpoint_file = "/mnt/ssd/sdd/khchoi/checkpoints/feature_models/mae_checkpoint-340.pth" # set your mae path
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint_model = checkpoint['model']

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg.missing_keys)
        trunc_normal_(model.head.weight, std=2e-5)

    device = torch.device(params.device)
    model.to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # # for MER2024 candidate 20000
    # import pandas as pd
    # filter_csv_path = "/home/amax/big_space/datasets/MER2024/dataset-process/my-process/candidate_20000.csv"
    # filter_df = pd.read_csv(filter_csv_path)
    # filter_names = set(filter_df['name'].tolist())

    # # Interrupt in the middle, continue
    # # feature_path = "/home/amax/big_space/datasets/MER2024/dataset-process/good_features/mae_399_UTT/"
    # feature_path = "/home/amax/big_space/datasets/MER2024/dataset-process/features_20000/mae_340_UTT/"
    # feats = os.listdir(feature_path)
    # for i in range(len(feats)):
    #     feats[i] = feats[i].split('.')[0]

    # extract embedding video by video
    vids = os.listdir(face_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        # # for MER2024 candidate 20000
        # if vid not in filter_names:
        #     print("continue")
        #     continue

        # # Interrupt in the middle, continue
        # if vid in feats:
        #     print("continue")
        #     continue

        print(f"Processing video '{vid}' ({i}/{len(vids)})...")

        # forward
        dataset = FaceDataset(vid, face_dir, transform=transform)
        if len(dataset) == 0:
            print("Warning: number of frames of video {} should not be zero.".format(vid))
            #import pdb; pdb.set_trace()
            embeddings, framenames = [], []
        else:
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=params.batch_size,
                                                      num_workers=10,
                                                      pin_memory=True)
            embeddings, framenames = extract(data_loader, model)

        # save results
        indexes = np.argsort(framenames)
        embeddings = embeddings[indexes]
        framenames = framenames[indexes]
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if params.feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(csv_file, embeddings)
        elif params.feature_level == 'BLK':
            embeddings = np.array(embeddings)
            if len(embeddings) == 0:
                embeddings = np.zeros((197, EMBEDDING_DIM))
            elif len(embeddings.shape) == 3:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)
            
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM,))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)


# EMER
# python -u extract_mae_embedding.py    --dataset='EMER' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'

# MER2023
# python -u extract_mae_embedding.py    --dataset='MER2023' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'

# MER2024
# python -u extract_mae_embedding.py    --dataset='MER2024' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'
# MER2024_20000
# python -u extract_mae_embedding.py    --dataset='MER2024_20000' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'

# DFEW
# python -u extract_mae_embedding.py    --dataset='DFEW' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'