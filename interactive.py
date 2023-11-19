from typing import Any, Dict, List
from loguru import logger
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import plotly.graph_objects as go
import random
from tqdm import tqdm
from operator import itemgetter
import streamlit as st
import datetime
import re
import os
from itertools import cycle

import torch
from torch.nn.parallel import DistributedDataParallel

from main_utils import parse_option
from models.bdetr import BeaUTyDETR
from src.grounding_evaluator import GroundingEvaluator
from src.joint_det_dataset import box2points
from train_dist_mod import TrainTester
from visualizer import make_bbox

st.set_page_config(
    page_title="EDA",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_NAME = "user"
DEFAULT_POINT_CLOUD_ID = 0
PLOTLY_PANEL_HEIGHT = 800
PLOTLY_COLORS = cycle([
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
])


def update_figure():
    with st.spinner("Creating a figure ..."):
        fig = go.Figure()

        # Point clouds
        fig.add_trace(
            go.Scatter3d(
                x=st.session_state.current_data["point_clouds"][:, 0],
                y=st.session_state.current_data["point_clouds"][:, 1],
                z=st.session_state.current_data["point_clouds"][:, 2],
                mode='markers',
                marker=dict(
                    size=1, color=st.session_state.current_data["og_color"]),
                name=f"Scene"
            )
        )

        # Bounding box (preds)
        bbox_coords = st.session_state.bbox_coords
        if bbox_coords is not None:
            for i_pred_box in range(len(bbox_coords)):
                fig.add_trace(
                    make_bbox(
                        bbox_coords[i_pred_box],
                        name=f"Prediction {i_pred_box + 1}",
                        color=next(PLOTLY_COLORS)
                    )
                )
                pass
            pass

        fig.update_layout(height=PLOTLY_PANEL_HEIGHT)
        st.plotly_chart(
            fig,
            use_container_width=True,
            height=PLOTLY_PANEL_HEIGHT
        )
        pass
    pass


def main():
    st.title("EDA Interactive Visualizer")

    args = parse_option()

    if "model" not in st.session_state:
        with st.spinner("Loading pretrain model ..."):
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "1111"

            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=datetime.timedelta(seconds=5400)
            )

            model = TrainTester.get_model(args)

            # Move model to devices
            assert torch.cuda.device_count() > 1
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                model
            ).cuda()

            # Convert state dict keys to non DDP types
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            state_dict: Dict[str, torch.Tensor] = checkpoint["model"]
            new_state_dict: Dict[str, torch.Tensor] = dict()

            for key, value in state_dict.items():
                new_key = key.replace("module.", "", 1)
                new_state_dict[new_key] = value
                pass

            model.load_state_dict(new_state_dict, strict=True)
            st.session_state.model: BeaUTyDETR = model
            pass
        pass

    if "dataset" not in st.session_state:
        with st.spinner("Loading dataset ..."):
            _, dataset = TrainTester.get_datasets(args, "fork")
            st.session_state.dataset = dataset
            pass
        pass

    if "point_cloud_id" not in st.session_state:
        st.session_state.point_cloud_id = DEFAULT_POINT_CLOUD_ID
        pass

    if "prompt_hist" not in st.session_state:
        st.session_state.prompt_hist = []
        pass

    if "current_data" not in st.session_state:
        st.session_state.current_data = st.session_state.dataset[st.session_state.point_cloud_id]
        st.session_state.bbox_coords = None
        pass

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = GroundingEvaluator(
            only_root=True,
            thresholds=[0.25, 0.5],
            topks=[1, 5, 10],
            prefixes=["last_"],
            filter_non_gt_boxes=args.butd_cls
        )
        pass

    # Construct streamlit
    if point_cloud_id := st.sidebar.number_input(
        label="Point Cloud ID",
        min_value=0,
        max_value=len(st.session_state.dataset),
        step=1
    ):
        if st.session_state.point_cloud_id != point_cloud_id:
            # Update point cloud data
            st.session_state.point_cloud_id = point_cloud_id
            st.session_state.current_data = st.session_state.dataset[st.session_state.point_cloud_id]

            st.session_state.bbox_coords = None
        pass

    if st.sidebar.button("Clear history"):
        st.session_state.prompt_hist.clear()
        st.session_state.bbox_coords = None
        pass

    if prompt := st.chat_input():
        # Inference
        with st.spinner("Predicting ..."):
            logger.info(f"Prompt: {prompt}")
            st.session_state.prompt_hist.append(prompt)

            with torch.no_grad():
                inputs = {
                    "point_clouds": torch.from_numpy(st.session_state.current_data["point_clouds"]).unsqueeze(0).cuda(),
                    "text": [prompt],
                    "train": False,
                    "det_boxes": torch.from_numpy(st.session_state.current_data["all_detected_boxes"]).unsqueeze(0).cuda(),
                    "det_bbox_label_mask": torch.from_numpy(st.session_state.current_data["all_detected_bbox_label_mask"]).unsqueeze(0).cuda(),
                    "det_class_ids": torch.from_numpy(st.session_state.current_data["all_detected_class_ids"]).unsqueeze(0).cuda()
                }

                end_points = st.session_state.model(inputs)

                for key in st.session_state.current_data:
                    assert (key not in end_points)
                    value = st.session_state.current_data[key]

                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value).unsqueeze(0).cuda()
                        pass

                    end_points[key] = value
                    pass

                for key in end_points:
                    if 'pred_size' in key:
                        end_points[key] = torch.clamp(
                            end_points[key],
                            min=1e-6
                        )
                        pass
                    pass

                outputs_to_return = st.session_state.evaluator.evaluate_bbox_by_pos_align(
                    end_points,
                    "last_"
                )

                bbox_coord = outputs_to_return["pred_bbox"]
                bbox_coord = bbox_coord.detach().cpu().squeeze(0).numpy()  # (10, 6)
                bbox_coord = box2points(bbox_coord)  # (10, 8, 3)
                st.session_state.bbox_coords = bbox_coord
                pass
            pass
        pass

    with st.sidebar.container():
        st.title("Prompt History")
        for prompt_hist in st.session_state.prompt_hist:
            with st.sidebar.chat_message(USER_NAME):
                st.markdown(prompt_hist)
                pass
            pass
        pass

    # Repaint panel
    update_figure()

    logger.info("Page Updated !")
    pass


if __name__ == "__main__":
    main()
    pass
