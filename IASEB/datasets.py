import json
import os
import cv2
import yaml
from .utils import xywh_to_corners


# Default empty paths — override via config file passed to --config
DATASET_PATHS = {}


def load_dataset_config(config_path):
    """Load dataset paths from a YAML config file."""
    global DATASET_PATHS
    with open(config_path, "r") as f:
        DATASET_PATHS.update(yaml.safe_load(f))


class HCSTVGDataloader:
    def __init__(self, args):
        self.referral_data_path = DATASET_PATHS[args.dataset]["referral"]
        self.video_dir = DATASET_PATHS[args.dataset]["video"]
        self.frame_step = args.frame_step
        self.args = args
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        if self.args.task_type == 'referral':
            self.referral_caption_data = json.load(open(self.referral_data_path, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = os.path.join(self.video_dir, entry['video_path'])
        cap = cv2.VideoCapture(video_path)

        # we only change from the original caption in the referral task
        if self.args.task_type == 'referral':
            if self.args.dataset == 'hcstvg1':
                entry["caption"] = self.referral_caption_data[entry["original_video_id"]][0]["phrases"][0]
            elif self.args.dataset == 'hcstvg2':
                entry["caption"] = self.referral_caption_data[entry["original_video_id"]]
        
        bbox_data = entry.get("trajectory", [])
        start_frame = entry.get('tube_start_frame', 0)
        end_frame = entry.get('tube_end_frame', len(bbox_data) - 1 + start_frame)

        frames_with_gt = []
        sampled_gt_bboxs = []

        # Seek to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        
        # Manually track the frame index for reliability
        current_frame_idx = start_frame

        while cap.isOpened():
            # Stop if we have processed all frames in the annotated range
            if current_frame_idx > end_frame:
                break
            
            ret, frame = cap.read()

            if not ret:
                break
            
            if (current_frame_idx - start_frame) % self.frame_step == 0:
                # We use bounding box index here because bbox annotations are in an array
                # As opposed to a dict with {frame_id: bbox}
                box_index = current_frame_idx - start_frame
                if 0 <= box_index < len(bbox_data):
                    bbox = xywh_to_corners(bbox_data[box_index])
                    frames_with_gt.append((frame, bbox, current_frame_idx))
                    sampled_gt_bboxs.append(bbox)
            
            # Increment our manual counter
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs


# ------------------------------------------------------------------------
# VidVRD dataset class verified to work with free form and referral format.
# Have not yet verified that it works for phrase format (if any, not sure).
# Ground truth boxes are in (0..H, 0..W) range and [x, y, w, h] format.
# Converts from [x,y,w,h] to [xmin, ymin, xmax, ymax] during loading.
# ------------------------------------------------------------------------

class VidVRDDataloader:
    def __init__(self, args):
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        self.frame_step = args.frame_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]
        bbox_data = entry.get("bbox", {})
        start_frame = entry.get("st_frame", 0)
        end_frame = entry.get("ed_frame", 0)

        cap = cv2.VideoCapture(video_path)
        frames_with_gt = []
        sampled_gt_bboxs = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        current_frame_idx = start_frame

        while cap.isOpened():
            if current_frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame_idx - start_frame) % self.frame_step == 0:
                bbox_sample = bbox_data.get(str(current_frame_idx), [0, 0, 0, 0])
                bbox_sample = xywh_to_corners(bbox_sample)
                frames_with_gt.append((frame, bbox_sample, current_frame_idx))
                sampled_gt_bboxs.append(bbox_sample)
            
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs

# --------------------
# VidSTG Dataset Class
# --------------------

class VidSTGDataloader:
    def __init__(self, args):
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        self.frame_step = args.frame_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]
        bbox_data = entry["bbox"]
        start_frame = entry.get("st_frame", 0)
        end_frame = entry.get("ed_frame", 0)

        cap = cv2.VideoCapture(video_path)
        frames_with_gt = []
        sampled_gt_bboxs = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        current_frame_idx = start_frame

        while cap.isOpened():
            if current_frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame_idx - start_frame) % self.frame_step == 0:
                bbox_sample = bbox_data.get(str(current_frame_idx), [0, 0, 0, 0])
                converted_bbox = xywh_to_corners(bbox_sample)
                frames_with_gt.append((frame, converted_bbox, current_frame_idx))
                sampled_gt_bboxs.append(converted_bbox)
            
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs


class MeViSBBoxDataloader():
    """
    Only loads frame and GT bbox every frame_step.
    """
    def __init__(self, args):
        print("Loading MeViS annotations...")
        self.data = []
        self.args = args
        self.boxes = json.load(open(DATASET_PATHS[args.dataset]["bbox"], 'r'))
        self.metadata = json.load(open(DATASET_PATHS[args.dataset]["metadata"], 'r'))

        for video_id, video_md in self.metadata["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "video_id": video_id,
                    "video_path": os.path.join(DATASET_PATHS[args.dataset]["video"], video_id), 
                    "caption": exp_data["exp"],
                    "anno_id": exp_data["anno_id"][0],
                    "obj_id": exp_data["obj_id"][0],
                    "gt_bboxs": {frame_id: xywh_to_corners(bbox) for frame_id, bbox in self.boxes["videos"][video_id]["expressions"][exp_id]["trajectory"].items()},
                })

        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path =  entry["video_path"]
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sampled_gt_bboxs = {}

        all_frames_np = []
        for idx, (frame_id, gt_bbox) in enumerate(entry["gt_bboxs"].items()):
            if idx % self.args.frame_step == 0:
                frame = cv2.imread(os.path.join(video_path, image_files[int(frame_id)]))
                assert frame is not None, f'ERROR: {frame_id} for {video_path} is None'
                all_frames_np.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sampled_gt_bboxs[frame_id] = gt_bbox

        entry["gt_bboxs"] = sampled_gt_bboxs

        return all_frames_np, entry


class ReferYouTubeVOSBBoxDataloader():
    def __init__(self, args):
        print("Loading Refer-YouTube-VOS metadata...")
        self.annotations = json.load(open(DATASET_PATHS[args.dataset]["bbox"], 'r'))
        self.data = []

        for video_id, video_md in self.annotations["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "exp_id": exp_id,
                    "video_id": video_id,
                    "video_path": os.path.join(DATASET_PATHS[args.dataset]["video"], video_id), 
                    "obj_id": exp_data["obj_id"],
                    "caption": exp_data["exp"],
                    "gt_bboxs": {frame_id: xywh_to_corners(bbox) for frame_id, bbox in exp_data["trajectory"].items()},

                })
        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        entry = self.data[idx]
        video_path =  entry["video_path"]
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sampled_gt_bboxs = {}

        # RVOS is presampled frame_step=5; we don't sample again
        all_frames_np = []
        for idx, (frame_id, gt_bbox) in enumerate(entry["gt_bboxs"].items()):
            frame = cv2.imread(os.path.join(video_path, image_files[idx]))
            assert frame is not None, f'ERROR: {frame_id} for {video_path} is None'
            all_frames_np.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sampled_gt_bboxs[frame_id] = gt_bbox

        entry["gt_bboxs"] = sampled_gt_bboxs

        return all_frames_np, entry

class STVGDataLoader:
    def __new__(cls, args):
        DATALOADER_MAP = {
            "hcstvg1": HCSTVGDataloader,
            "hcstvg2": HCSTVGDataloader,
            "vidstg": VidSTGDataloader,
            "vidvrd": VidVRDDataloader,
            "mevis": MeViSBBoxDataloader,
            "rvos": ReferYouTubeVOSBBoxDataloader
        }
        return DATALOADER_MAP[args.dataset](args)
