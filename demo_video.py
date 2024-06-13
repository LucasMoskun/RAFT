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
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + '_video_depth.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        while raw_video.isOpened():
            with torch.no_grad():
                re1, frame1 = raw_video.read()
                re2, frame2 = raw_video.read()
                if not re1 or not re2:
                    break
                img1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
                frame1 = img1[None].to(DEVICE)
                img2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
                frame2 = img2[None].to(DEVICE)
                padder = InputPadder(frame1.shape)
                frame1, frame2 = padder.pad(frame1, frame2)
                flw_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
                img = frame1[0].permute(1,2,0).cpu().numpy()
                flo = flow_up[0].permute(1,2,0).cpu().numpy()
                flo = flow_viz.flow_to_image(flo)

                out.write((flo[:, :, [2,1,0]] * 255).astype(np.uint8))

            
        
        raw_video.release()
        out.release()