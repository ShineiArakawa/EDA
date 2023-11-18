import os
import pickle
from loguru import logger
import numpy as np
import plotly.graph_objects as go

from src.joint_det_dataset import Joint3DDataset, box2points
from train_dist_mod import TrainTester
from main_utils import parse_option

def make_bbox(
    box_corner: np.ndarray,
    name: str,
    color: str = "#00ff00",
    visible: bool = True
):
    line_pairs = [
        [0, 1],
        [0, 2],
        [0, 4],
        [7, 3],
        [7, 5],
        [7, 6],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [1, 5],
        [2, 6]
    ]
    line_x = []
    line_y = []
    line_z = []
    
    for i_line_pair in range(len(line_pairs)):
        for i in range(2):
            line_x.append(box_corner[line_pairs[i_line_pair][i]][0])
            line_y.append(box_corner[line_pairs[i_line_pair][i]][1])
            line_z.append(box_corner[line_pairs[i_line_pair][i]][2])
        line_x.append(None)
        line_y.append(None)
        line_z.append(None)
        pass

    return go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode='lines',
        name=name,
        line=dict(
            color=color
        ),
        visible=visible
    )

def main():
    args = parse_option()
    
    # ================= Load dataset ================= 
    dataset: Joint3DDataset = None
    dataset_pickle_path = 'dataset_for_visualize.pkl'
    if not os.path.exists(dataset_pickle_path):
        _, dataset = TrainTester.get_datasets(args)
        # Save dataset instance as pickle for fast loading. 
        # This operation is executed once and the dataset is loaded from the pickle file from next time.
        with open(dataset_pickle_path, 'wb') as file:
            pickle.dump(dataset, file)
            pass
        pass
    
    if dataset is None:
        with open(dataset_pickle_path, 'rb') as file:
            dataset = pickle.load(file)
            pass
        
    
    
    # ================= Get data ================= 
    one_data = dataset.__getitem__(0)
    
    # Get bbox coordinate
    target_id = one_data["target_id"]
    all_boxes_points = box2points(one_data["all_detected_boxes"][..., :6])
    
    bbox_corners = all_boxes_points[target_id]
    
    tmp_anchor_ids: np.ndarray = one_data['anchor_ids']
    anchor_ids = np.array(
        tmp_anchor_ids.tolist() + [-1] * (10 - len(tmp_anchor_ids))
    ).astype(int)
    anchor_bbox_corners = all_boxes_points[[
        i.item() for i in anchor_ids if i != -1
    ]]
    
    tmp_distractor_ids: np.ndarray = one_data['distractor_ids']
    distractor_ids = np.array(
        tmp_distractor_ids.tolist() + [-1] * (10 - len(tmp_distractor_ids))
    ).astype(int)
    distractor_bbox_corners = all_boxes_points[[
        i.item() for i in distractor_ids if i != -1
    ]]
    
    logger.info(f'one_data["point_clouds"].shape                : {one_data["point_clouds"].shape}')
    logger.info(f'one_data["og_color"].shape                    : {one_data["og_color"].shape}')
    logger.info(f'one_data["utterances"]                        : {one_data["utterances"]}')
    logger.info(f'one_data["all_detected_boxes"].shape          : {one_data["all_detected_boxes"].shape}')
    logger.info(f'one_data["all_detected_bbox_label_mask"].shape: {one_data["all_detected_bbox_label_mask"].shape}')
    logger.info(f'one_data["all_detected_class_ids"].shape      : {one_data["all_detected_class_ids"].shape}')
    logger.info(f'one_data["target_id"]                         : {one_data["target_id"]}')
    logger.info(f'one_data["target_name"]                       : {one_data["target_name"]}')
    logger.info(f'len(anchor_bbox_corners)                      : {len(anchor_bbox_corners)}')
    logger.info(f'len(distractor_bbox_corners)                  : {len(distractor_bbox_corners)}')
    logger.info(f'bbox_corners.shape                            : {bbox_corners.shape}')

    # ================= Visualize =================
    graph_objects = []
    graph_objects.append(
        go.Scatter3d(
            x=one_data["point_clouds"][:,0],
            y=one_data["point_clouds"][:,1],
            z=one_data["point_clouds"][:,2],
            mode='markers',
            marker=dict(size=1, color=one_data["og_color"]),
            name="Scene"
        )
    )
    graph_objects.append(make_bbox(bbox_corners, "bbox"))
    
    fig = go.Figure(
        data=graph_objects
    )
    fig.show()

if __name__ == "__main__":
    main()
    pass