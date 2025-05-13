# @Time    : 6/24/23 9:33 AM
# @Author  : bbbdbbb
# @File    : extract_maeVideo_embedding.py
# @Description : load maeVideo model to extract video feature embedding

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
from timm.models import create_model

import sys

sys.path.append('../../')
#import config
from dataset import FaceDataset

from maeVideo import models_vit
from collections import OrderedDict
from maeVideo.modeling_finetune import vit_large_patch16_224
from maeVideo.dataset_MER import train_data_loader, test_data_loader


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='EMER', help='input dataset')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--pretrain_model', type=str, default='VoxCeleb_ckp49', help='pth of pretrain MAE model')
    parser.add_argument('--feature_name', type=str, default='VoxCeleb_ckp49', help='pth of pretrain MAE model')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--nb_classes', default=6, type=int, help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=1, type=int)



    params = parser.parse_args()

    print(f'==> Extracting maeVideo embedding...')
    # face_dir = config.PATH_TO_RAW_FACE[params.dataset]
    # save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'{params.feature_name}_{params.feature_level[:3]}')
    if params.dataset == "MER2023":
        face_dir = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/dataset-process/pyfeat_face/crop_images/"
        save_dir = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/features/AIHUB_maeV_399_UTT"
        list_file = "/mnt/ssd/sdd/khchoi/datasets/AIHUB_Kor/faceresult/list_files/MER2023_NCEV_KHTEST.txt"
    elif params.dataset == "EMER":
        face_dir = "/home/amax/big_space/datasets/MER2024/EMER/all_face"
        save_dir = "/home/amax/big_space/datasets/MER2024/EMER/features_tmp/maeV_199_UTT"
        list_file = "/home/amax/big_space/datasets/list_files/EMER_332_NCE.txt"
    elif params.dataset == "MER2024":
        face_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/all_face"
        save_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/features_tmp/maeV_199_UTT"
        list_file = "/home/amax/big_space/datasets/list_files/MER2024_12065_NCE.txt"
    elif params.dataset == "MER2024_20000":
        face_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/all_face"
        save_dir = "/home/amax/big_space/datasets/MER2024/dataset-process/features_20000_tmp/maeV_199_UTT"
        list_file = "/home/amax/big_space/datasets/list_files/MER2024_candidate_20000.txt"
    elif params.dataset == "DFEW":
        face_dir = "/home/amax/big_space/datasets/DFEW/dataset-process/openface_face"
        save_dir = "/home/amax/big_space/datasets/DFEW/dataset-process/features_tmp/maeV_199_UTT"
        list_file = "/home/amax/big_space/datasets/list_files/DFEW_set_1_train.txt"
        # list_file = "/home/amax/big_space/datasets/list_files/DFEW_set_1_test.txt"
    if not os.path.exists(save_dir): os.makedirs(save_dir)


    # load model
    model = vit_large_patch16_224()

    if True:
        # checkpoint_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'maeVideo', params.pretrain_model + '.pth')
        checkpoint_file = "/mnt/ssd/sdd/khchoi/checkpoints/feature_models/maeVideo_ckp399.pth"
        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        checkpoint_model = None
        for model_key in 'model|module'.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (16 // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (16 // model.patch_embed.tubelet_size)) ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, 16 // model.patch_embed.tubelet_size, orig_size, orig_size,
                                                embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, 16 // model.patch_embed.tubelet_size, new_size,
                                                                    new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        load_state_dict(model, checkpoint_model, prefix='')
        # trunc_normal_(model.head.weight, std=2e-5)

    device = torch.device(params.device)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[1])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # dataset
    dataset = test_data_loader(list_file, face_dir)
    # dataset = train_data_loader(list_file, face_dir)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=10,
        drop_last=True,
    )

    i = 1
    vids = len(data_loader)
    for images, video_name in data_loader:
        print(f"Processing video ' ({i}/{vids})...")
        i = i + 1
        images = images.to(device)
        embedding = model(images)

        print("embedding :", embedding.shape)
        embedding = embedding.cpu().detach().numpy()

        # save results
        EMBEDDING_DIM = max(-1, np.shape(embedding)[-1])

        video_name = video_name[0]

        csv_file = os.path.join(save_dir, f'{video_name}.npy')
        if params.feature_level == 'FRAME':
            embedding = np.array(embedding).squeeze()
            if len(embedding) == 0:
                embedding = np.zeros((1, EMBEDDING_DIM))
            elif len(embedding.shape) == 1:
                embedding = embedding[np.newaxis, :]
            np.save(csv_file, embedding)
        elif params.feature_level == 'BLK':
            embedding = np.array(embedding)
            if len(embedding) == 0:
                embedding = np.zeros((257, EMBEDDING_DIM))
            elif len(embedding.shape) == 3:
                embedding = np.mean(embedding, axis=0)
            np.save(csv_file, embedding)
        else:
            embedding = np.array(embedding).squeeze()
            if len(embedding) == 0:
                embedding = np.zeros((EMBEDDING_DIM,))
            elif len(embedding.shape) == 2:
                embedding = np.mean(embedding, axis=0)
            print("csv_file: ", csv_file)
            print("embedding: ", embedding)
            np.save(csv_file, embedding)

# MER2023
# python -u extract_maeVideo_embedding.py    --dataset='MER2023' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# EMER
# python -u extract_maeVideo_embedding.py    --dataset='EMER' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# MER2024
# python -u extract_maeVideo_embedding.py    --dataset='MER2024' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'
# MER2024_20000
# python -u extract_maeVideo_embedding.py    --dataset='MER2024_20000' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# DFEW
# python -u extract_maeVideo_embedding.py    --dataset='DFEW' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'