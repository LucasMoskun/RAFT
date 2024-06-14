import sys
sys.path.append('core')

import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from PIL import Image
import glob


from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


def process_sequence(sequence_dir, model, DEVICE, args, outdir):
    image_files = sorted(glob.glob(os.path.join(sequence_dir, '*.png')))
    if not image_files:
        return

    basename = os.path.basename(sequence_dir)
    output_dir = os.path.join(outdir, basename)
    os.makedirs(output_dir, exist_ok=True)
    png_output_path = os.path.join(output_dir, basename + '_optical_flow_png')
    os.makedirs(png_output_path, exist_ok=True)
    output_path = os.path.join(output_dir, basename + '_optical_flow.mp4')

    frame_height, frame_width = cv2.imread(image_files[0]).shape[:2]
    frame_rate = 30  # Assuming a standard frame rate; adjust as necessary
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    prev_frame = None
    frame_count = 0
    for image_file in image_files:
        curr_frame = cv2.imread(image_file)
        if prev_frame is None:
            prev_frame = curr_frame
            continue

        with torch.no_grad():
            img1 = torch.from_numpy(prev_frame).permute(2, 0, 1).float()
            frame1 = img1[None].to(DEVICE)
            img2 = torch.from_numpy(curr_frame).permute(2, 0, 1).float()
            frame2 = img2[None].to(DEVICE)
            padder = InputPadder(frame1.shape)
            frame1, frame2 = padder.pad(frame1, frame2)
            _, flow_up = model(frame1, frame2, iters=20, test_mode=True)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            flo = flow_viz.flow_to_image(flo)

            out.write((flo[:, :, [2,1,0]] * 255).astype(np.uint8))

            png_filename = os.path.join(png_output_path, '{:04d}.png'.format(frame_count))
            success = cv2.imwrite(png_filename, flo)
            if not success:
                print('Failed to write', png_filename)

            frame_count += 1
            prev_frame = curr_frame

    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-dir', type=str, help="Directory containing directories of image sequences")
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    sequence_dirs = [os.path.join(args.sequence_dir, d) for d in os.listdir(args.sequence_dir) if os.path.isdir(os.path.join(args.sequence_dir, d))]

    for sequence_dir in sequence_dirs:
        process_sequence(sequence_dir, model, DEVICE, args, args.outdir)