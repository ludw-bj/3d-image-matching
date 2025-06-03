import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from transformers import AutoImageProcessor, AutoModel


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def get_global_desc(fnames, device = torch.device('cpu')):
    processor = AutoImageProcessor.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = AutoModel.from_pretrained('/kaggle/input/dinov2/pytorch/base/1')
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            dino_mac = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=1, p=2)
        global_descs_dinov2.append(dino_mac.detach().cpu())
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2

def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs

def get_image_pairs_shortlist(fnames,
                              sim_th = 0.59, # should be strict
                              min_pairs = 20,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    """ Core logic for image pair selection

    Parameters:
        sim_th: Similarity threshold (on Euclidean distance) to consider two descriptors as matching.
        min_pairs: Minimum number of matches each image should have.
        exhaustive_if_less: If number of images is small (≤ threshold), fallback to exhaustive matching.

    Returns: A list of (i, j) index pairs of potentially matching images.
    """
    num_imgs = len(fnames)
    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    descs = get_global_desc(fnames, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list