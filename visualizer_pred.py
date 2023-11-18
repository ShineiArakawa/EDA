from loguru import logger
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import plotly.graph_objects as go

import torch
from src.joint_det_dataset import box2points

from visualizer import make_bbox

def parse_args():
    parser = ArgumentParser(
        description="Visualize predictions of EDA",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "pred_file",
        type=str
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="last"
    )
    parser.add_argument(
        "--max-n-data",
        type=int,
        default=10
    )
    return parser.parse_args()


def read_prediction_file(file_path: str, prefix: str):
    logger.info(f"Load predictions from {file_path}")
    logger.info(f"Prefix is '{prefix}'")
    
    data = torch.load(file_path)[prefix + "_"]
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].numpy()
    
    return data


def main():
    args = parse_args()

    # Load predictions file
    preds = read_prediction_file(args.pred_file, args.prefix)
    
    # Log data shapes
    logger.info(f"preds['gt_bbox'].shape    : {preds['gt_bbox'].shape}")
    logger.info(f"preds['pred_bbox'].shape  : {preds['pred_bbox'].shape}")
    logger.info(f"preds['ious'].shape       : {preds['ious'].shape}")
    logger.info(f"preds['point_cloud'].shape: {preds['point_cloud'].shape}")
    logger.info(f"preds['orig_color'].shape : {preds['orig_color'].shape}")
    logger.info(f"len(preds['utterances'])  : {len(preds['utterances'])}")
    logger.info(f"len(preds['target_name']) : {len(preds['target_name'])}")
    
    preds['gt_bbox'] = box2points(preds['gt_bbox'][..., :6])
    pred_bbox_shape = preds['pred_bbox'].shape
    preds['pred_bbox'] = box2points(preds['pred_bbox'].reshape(-1, 6)).reshape(pred_bbox_shape[0], pred_bbox_shape[1], 8, 3)
    
    # Plot
    fig = go.Figure()
    
    n_data = min(args.max_n_data, len(preds['utterances']))
    for i_data in range(n_data):
        n_plots_per_data = 0
        
        # Point clouds
        fig.add_trace(
            go.Scatter3d(
                x=preds['point_cloud'][i_data, :, 0],
                y=preds['point_cloud'][i_data, :, 1],
                z=preds['point_cloud'][i_data, :, 2],
                mode='markers',
                marker=dict(size=1, color=preds['orig_color'][i_data]),
                name=f"Scene",
                visible=True if i_data == 0 else False
            )
        )
        n_plots_per_data += 1
        
        # Bounding box (GT)
        fig.add_trace(
            make_bbox(
                preds['gt_bbox'][i_data],
                name=f"Ground truth (target name: '{preds['target_name'][i_data]}', utterances: '{preds['utterances'][i_data]}')",
                color="#FF0000",
                visible=True if i_data == 0 else False
            )
        )
        n_plots_per_data += 1
        
        # Bounding box (preds)
        for i_pred_box in range(len(preds['pred_bbox'][i_data])):
            fig.add_trace(
                make_bbox(
                    preds['pred_bbox'][i_data][i_pred_box],
                    name=f"Prediction {i_pred_box + 1} (IOU={preds['ious'][i_data][i_pred_box]})",
                    color="#00ff00",
                    visible=True if i_data == 0 else False
                )
            )
            n_plots_per_data += 1
            pass


    updatemenus_button_opts = []
    for i_data in range(n_data):
        visibility = [False for _ in range(n_data * n_plots_per_data)]
        
        for i_plot in range(n_plots_per_data):
            visibility[i_data * n_plots_per_data + i_plot] = True
            pass
        
        updatemenus_button_opts.append(dict(
            label=f"Prediction (i={i_data})",
            method="update",
            args=[{"visible": visibility}]
        ))
        pass
    
    updatemenus = [{
        "active": 0,
        "type": "dropdown",
        "buttons": updatemenus_button_opts
    }]

    fig.update_layout(updatemenus=updatemenus)
    fig.show()
    pass


if __name__ == "__main__":
    main()
    pass
