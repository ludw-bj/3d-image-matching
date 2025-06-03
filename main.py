import os
import argparse
import gc
from collections import defaultdict
from time import time
from copy import deepcopy
import pycolmap

from utils.io_utils import load_predictions  # Refactored from your logic
from utils.colmap_utils import import_into_colmap
from utils.feature_extraction import detect_aliked, match_with_lightglue
from utils.shortlist import get_image_pairs_shortlist
from utils.cluster import hierarchical, dbscan, optics  # Import actual functions

CLUSTER_METHODS = {
    "hierarchical": hierarchical.cluster_with_hierarchical,
    "dbscan": dbscan.cluster_with_DBSCAN,
    "optics": optics.cluster_with_optics
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action='store_false', help='Training mode')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/image-matching-challenge-2025')
    parser.add_argument('--workdir', type=str, default='/kaggle/working/result/')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--cluster_method', type=str, choices=CLUSTER_METHODS.keys(), default='hierarchical')

    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main(args):
    os.makedirs(args.workdir, exist_ok=True)

    predictions_by_dataset = load_predictions(args.data_dir, args.is_train)
    timings = defaultdict(list)
    mapping_result_strs = []

    for dataset, predictions in predictions_by_dataset.items():
        if args.datasets and dataset not in args.datasets:
            continue

        print(f'\nProcessing dataset "{dataset}"')
        images_dir = os.path.join(args.data_dir, 'train' if args.is_train else 'test', dataset)
        images = [os.path.join(images_dir, p.filename) for p in predictions]
        if args.max_images:
            images = images[:args.max_images]

        try:
            # CLUSTERING
            cluster_func = CLUSTER_METHODS[args.cluster_method]

            t = time()
            components, cluster_map = cluster_func(images)
            timings['clustering'].append(time() - t)
            print(f"Clustering done in {time() - t:.2f}s | Clusters: {len(components)}")

            filename_to_index = {p.filename: idx for idx, p in enumerate(predictions)}
            clusters = defaultdict(list)
            for fname, cid in cluster_map.items():
                clusters[cid].append(fname)

            for img_path in images:
                prediction_index = filename_to_index[os.path.basename(img_path)]
                cluster_id = cluster_map[img_path]
                predictions[prediction_index].cluster_index = cluster_id if cluster_id != -1 else None

            for cluster_id, cluster_images in clusters.items():
                if cluster_id == -1 or len(cluster_images) < 3:
                    print(f"Skipping cluster {cluster_id} (invalid or too small).")
                    for img_path in cluster_images:
                        prediction_index = filename_to_index[img_path.split('/')[-1]]
                        predictions[prediction_index].cluster_index = None
                        predictions[prediction_index].rotation = None
                        predictions[prediction_index].translation = None
                    continue

                print(f"== Cluster {cluster_id} with {len(cluster_images)} images ==")
                cluster_dir = os.path.join(args.workdir, 'featureout', dataset, f'cluster_{cluster_id}')
                os.makedirs(cluster_dir, exist_ok=True)
                cluster_db_path = os.path.join(cluster_dir, 'colmap.db')
                cluster_output_path = os.path.join(cluster_dir, 'colmap_rec_aliked')

                # Shortlist
                t = time()
                index_pairs = get_image_pairs_shortlist(cluster_images, device=args.device)
                timings['shortlist'].append(time() - t)

                # Feature Detection
                t = time()
                detect_aliked(cluster_images, cluster_dir, 4096, device=args.device)
                timings['feature_detection'].append(time() - t)

                # Feature Matching
                t = time()
                match_with_lightglue(cluster_images, index_pairs, feature_dir=cluster_dir, device=args.device)
                timings['feature_matching'].append(time() - t)

                # Geometric Verification
                if os.path.isfile(cluster_db_path):
                    os.remove(cluster_db_path)
                import_into_colmap(images_dir, cluster_dir, cluster_db_path)
                t = time()
                pycolmap.match_exhaustive(cluster_db_path)
                timings['RANSAC'].append(time() - t)

                # Mapping
                t = time()
                mapper_options = pycolmap.IncrementalPipelineOptions()
                mapper_options.min_model_size = 3  # Lower threshold to allow small clusters
                mapper_options.max_num_models = 25
                os.makedirs(cluster_output_path, exist_ok=True)

                t = time()
                maps = pycolmap.incremental_mapping(
                    database_path=cluster_db_path,
                    image_path=images_dir,
                    output_path=cluster_output_path,
                    options=mapper_options)
                timings['Reconstruction'].append(time() - t)

                for map_id, m in maps.items():
                    for img in m.images.values():
                        idx = filename_to_index[img.name]
                        predictions[idx].rotation = deepcopy(img.cam_from_world.rotation.matrix())
                        predictions[idx].translation = deepcopy(img.cam_from_world.translation)

                print(f"Reconstruction success: {sum(len(m.images) for m in maps.values())} images")

            gc.collect()
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            # raise e
            mapping_result_str = f'Dataset "{dataset}" -> Failed!'
            mapping_result_strs.append(mapping_result_str)
            print(mapping_result_str)
            continue

    # Save submission
    save_submission(predictions_by_dataset, args.is_train)


def save_submission(preds_by_dataset, is_train):
    def array_to_str(array): return ';'.join([f"{x:.09f}" for x in array])
    def none_to_str(n): return ';'.join(['nan'] * n)

    sub_file = '/kaggle/working/submission.csv'
    with open(sub_file, 'w') as f:
        if is_train:
            f.write('dataset,scene,image,rotation_matrix,translation_vector\n')
        else:
            f.write('image_id,dataset,scene,image,rotation_matrix,translation_vector\n')

        for ds in preds_by_dataset:
            for p in preds_by_dataset[ds]:
                scene = f'cluster{p.cluster_index}' if p.cluster_index is not None else 'outliers'
                R = none_to_str(9) if p.rotation is None else array_to_str(p.rotation.flatten())
                t = none_to_str(3) if p.translation is None else array_to_str(p.translation)
                if is_train:
                    f.write(f'{p.dataset},{scene},{p.filename},{R},{t}\n')
                else:
                    f.write(f'{p.image_id},{p.dataset},{scene},{p.filename},{R},{t}\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
