import os
from tqdm import tqdm
import h5py
import torch
import kornia.feature as KF
import kornia as K
from lightglue import ALIKED

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def detect_aliked(img_fnames,
                  feature_dir = '.featureout',
                  num_features = 4096,
                  resize_to = 1024,
                  device=torch.device('cpu')):
    dtype = torch.float32 # ALIKED has issues with float16
    extractor = ALIKED(max_num_keypoints=num_features, detection_threshold=0.3, resize=resize_to).eval().to(device, dtype)
    # if not os.path.isdir(feature_dir):
    #     os.makedirs(feature_dir)
    # if os.path.isdir(feature_dir):
    #     clear_feature_dir(feature_dir)
    os.makedirs(feature_dir, exist_ok=True)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                image0 = load_torch_image(img_path, device=device).to(dtype)
                feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
                kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    f_kp.close()
    f_desc.close()
    return

def match_with_lightglue(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=20,verbose=True):
    lg_matcher = KF.LightGlueMatcher("aliked", {"width_confidence": -1,
                                                "depth_confidence": -1,
                                                 "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
        h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
            with torch.inference_mode():
                dists, idxs = lg_matcher(desc1,
                                         desc2,
                                         KF.laf_from_center_scale_ori(kp1[None]),
                                         KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs)  == 0:
                continue
            n_matches = len(idxs)
            if verbose:
                print (f'{key1}-{key2}: {n_matches} matches')
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    f_kp.close()
    f_desc.close()
    f_match.close()
    return