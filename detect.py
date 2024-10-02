import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

random.seed(1)

# Helper function to smooth bounding box centers using exponential moving average
def smooth_center_position(new_center, prev_center, alpha=0.5):
    """
    Smooths the center position using an exponential moving average (EMA).
    
    Args:
        new_center (tuple): New center position (x, y).
        prev_center (tuple): Previous center position (x, y).
        alpha (float): Smoothing factor. Higher alpha values give more weight to new positions.
        
    Returns:
        tuple: Smoothed center position.
    """
    smoothed_x = int(alpha * new_center[0] + (1 - alpha) * prev_center[0])
    smoothed_y = int(alpha * new_center[1] + (1 - alpha) * prev_center[1])
    return smoothed_x, smoothed_y
            
# Variables to store the previous center positions for smoothing
prev_center = None


def detect(save_img=False):
    global prev_center            

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh


            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Lists to store person bounding boxes and their respective center points
                person_bboxes = []
                person_centers = []

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        # Check if detected class is 'person'
                        if names[int(cls)] == 'person':
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            bbox_center_x, bbox_center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the person

                            # Append bounding box and center point
                            person_bboxes.append([x1, y1, x2, y2])
                            person_centers.append((bbox_center_x, bbox_center_y))
                # Ensure at least one person was detected
                if person_bboxes:
                    # Select the person to focus on based on a specific criterion
                    # Option 1: Largest person (based on bounding box area)
                    person_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in person_bboxes]
                    largest_person_idx = np.argmax(person_areas)  # Index of the largest person

                    # Option 2: Closest to the frame center
                    frame_center_x, frame_center_y = im0.shape[1] // 2, im0.shape[0] // 2
                    distances_to_center = [((cx - frame_center_x) ** 2 + (cy - frame_center_y) ** 2) ** 0.5 for cx, cy in person_centers]
                    closest_person_idx = np.argmin(distances_to_center)  # Index of the closest person

                    # Choose the desired person selection method (e.g., largest person)
                    selected_person_idx = closest_person_idx  # Change to `closest_person_idx` if desired

                    # Get the bounding box of the selected person
                    x1, y1, x2, y2 = person_bboxes[selected_person_idx]
                    bbox_center_x, bbox_center_y = person_centers[selected_person_idx]  # Center of the selected person's bbox

                    # Select the target person (e.g., the first person detected)
                    target_center = person_centers[selected_person_idx]

                    # Smooth the center position using EMA
                    if prev_center is None:
                        prev_center = target_center  # Initialize with the first frame's position
                    smoothed_center = smooth_center_position(target_center, prev_center, alpha=0.2)

                    # Update previous center with the current smoothed position
                    prev_center = smoothed_center

                    # Set a zoom factor (e.g., 1.5x or 2x zoom)
                    zoom_factor = 2

                    # Get original frame dimensions
                    frame_height, frame_width, _ = im0.shape

                    # Calculate the size of the zoomed area (width and height after zooming)
                    zoomed_width = frame_width // zoom_factor
                    zoomed_height = frame_height // zoom_factor
                    
                    zoomed_width = frame_width * 2 // 3
                    zoomed_height = frame_height * 2 // 3

                    # Calculate the cropping coordinates to keep the person at the center (stabilization version)
                    crop_x1 = max(0, smoothed_center[0] - zoomed_width // 2)
                    crop_y1 = max(0, smoothed_center[1] - zoomed_height // 2)
                    if crop_x1 == 0:
                      crop_x2=zoomed_width
                    else:
                      crop_x2 = min(frame_width, smoothed_center[0] + zoomed_width // 2)
                    if crop_y1 == 0:
                      crop_y2=zoomed_height
                    else:
                      crop_y2 = min(frame_height, smoothed_center[1] + zoomed_height // 2)

#                    # Calculate the cropping coordinates to keep the person at the center
#                    crop_x1 = max(0, bbox_center_x - zoomed_width // 2)
#                    crop_y1 = max(0, bbox_center_y - zoomed_height // 2)
#                    if crop_x1 == 0:
#                      crop_x2=zoomed_width
#                    else:
#                      crop_x2 = min(frame_width, bbox_center_x + zoomed_width // 2)
#                    if crop_y1 == 0:
#                      crop_y2=zoomed_height
#                    else:
#                      crop_y2 = min(frame_height, bbox_center_y + zoomed_height // 2)

                    # Crop the area centered on the selected person
                    cropped_frame = im0[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Resize the cropped frame back to original size to simulate zoom
                    zoomed_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

                    # Update the current frame to show the zoomed frame
                    im0 = zoomed_frame

#                            # Set a zoom factor (e.g., 1.5x or 2x zoom)
#                            zoom_factor = 2
#
#                            # Get original frame dimensions
#                            frame_height, frame_width, _ = im0.shape
#
#                            # Calculate the size of the zoomed area (width and height after zooming)
#                            zoomed_width = frame_width // zoom_factor
#                            zoomed_height = frame_height // zoom_factor
#
#                            # Calculate the cropping coordinates to keep the person at the center
#                            crop_x1 = max(0, bbox_center_x - zoomed_width // 2)
#                            crop_y1 = max(0, bbox_center_y - zoomed_height // 2)
#                            crop_x2 = min(frame_width, bbox_center_x + zoomed_width // 2)
#                            crop_y2 = min(frame_height, bbox_center_y + zoomed_height // 2)
#
#                            # Crop the area centered on the person
#                            cropped_frame = im0[crop_y1:crop_y2, crop_x1:crop_x2]
#
#                            # Resize the cropped frame back to original size to simulate zoom
#                            zoomed_frame = cv2.resize(cropped_frame, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
#
#                            # Update the current frame to show the zoomed frame
#                            im0 = zoomed_frame


            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        #im_crop=im0[x1f:x2f, y1f:y2f]
                        #print('%s, %s, %s, %s\n' % (x1f, y1f, x2f, y2f))
                        #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (im_crop.shape[1], im_crop.shape[0]))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #vid_writer.write(im_crop)
                    vid_writer.write(im0)

    if save_txt or save_img:
        print(f" The output with the result is saved in: {save_path}")
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
