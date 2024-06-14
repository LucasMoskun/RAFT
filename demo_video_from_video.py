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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    args = parser.parse_args()
    
    margin_width = 50

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()


    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = os.listdir(args.video_path)
        filenames = [os.path.join(args.video_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print('Progress {:}/{:},'.format(k+1, len(filenames)), 'Processing', filename)
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        output_width = frame_width
        
        filename = os.path.basename(filename)
        basename = filename[:filename.rfind('.')]
        output_dir = os.path.join(args.outdir, basename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, basename + '_optical_flow.mp4')
        png_output_path = os.path.join(output_dir, basename + '_optical_flow_png')
        if not os.path.exists(png_output_path):
            os.makedirs(png_output_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        frame_count = 0 
        prev_frame = None
        while raw_video.isOpened():
            with torch.no_grad():
                ret, curr_frame = raw_video.read()
                if not ret:
                    break
                if prev_frame is None:
                    prev_frame = curr_frame
                    continue
                img1 = torch.from_numpy(prev_frame).permute(2, 0, 1).float()
                frame1 = img1[None].to(DEVICE)
                img2 = torch.from_numpy(curr_frame).permute(2, 0, 1).float()
                frame2 = img2[None].to(DEVICE)
                padder = InputPadder(frame1.shape)
                frame1, frame2 = padder.pad(frame1, frame2)
                flw_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
                img = frame1[0].permute(1,2,0).cpu().numpy()
                flo = flow_up[0].permute(1,2,0).cpu().numpy()
                flo = flow_viz.flow_to_image(flo)

                out.write((flo[:, :, [2,1,0]] * 255).astype(np.uint8))

                png_filename = os.path.join(png_output_path, '{:04d}.png'.format(frame_count))
                success = cv2.imwrite(png_filename, flo)
                if not success:
                    print('Failed to write', png_filename)
                frame_count += 1
            prev_frame = curr_frame

            
        
        raw_video.release()
        out.release()
