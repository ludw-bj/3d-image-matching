from hloc.utils.database import COLMAPDatabase
from hloc.utils.colmap import add_keypoints, add_matches

def import_into_colmap(img_dir, feature_dir ='.featureout', database_path = 'colmap.db'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, '', 'simple-pinhole', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
    return