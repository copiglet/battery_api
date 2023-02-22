import argparse
import time
from pathlib import Path
import os, sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import shutil
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import pmxevent as pevent 

import json



class HvDetect():
    def __init__(self, save_img, debug_print, image_path, web_debug):
        self.weights, self.view_img, self.save_txt, self.imgsz, self.trace = './models/best.pt', False, False, 640, True
        self.save_img = save_img
        self.debug_print = debug_print
        self.image_path = image_path
        self.web_debug = web_debug

        print('==========DETECT-CONFIG===========')
        print(f" - weights:{self.weights}")
        print(f" - view_img:{self.view_img}")
        print(f" - save_txt:{self.save_txt}")
        print(f" - imgsz:{self.imgsz}")
        print(f" - trace:{self.trace}")
        print("==========================")

        # Initialize
        if self.debug_print:
            set_logging()
        else:
            set_logging(2)

        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA


        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, 640)
            
        if self.half:
            self.model.half()
        
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # colors = [80,205,180,120]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1


    def setconfig(self, save_img, debug_print, image_path, web_debug):
        self.save_img = save_img
        self.debug_print = debug_print
        self.image_path = image_path
        self.web_debug = web_debug

    # return code, detail
    #
    def run_detect(self, input_id, pt_path = '', fn_name = ''):
        image_pt_path = self.image_path if pt_path == '' else pt_path
        ret_json = {"id": input_id, "ng": 1, "detail": []}
        colors = [80,205,180,120]
        if fn_name == '':
            input_path = os.path.join(image_pt_path, input_id)
        else:
            input_path = os.path.join(image_pt_path, input_id, fn_name)

        print('run_detect :' + input_path)
        if os.path.isdir(input_path):
            print(f"run for cellid folder: {input_path}")
        elif os.path.isfile(input_path):
            print(f"run for file : {input_path}")
        else:
            print(404, {"error": f"input path fail: {input_path}"})
            return 404, {"error": f"input path fail: {input_path}"} 

        im_path = input_path
        save_dir = Path(increment_path(Path('./runs/detect') / im_path.split('/')[-1], exist_ok=True))  # increment run
        if self.save_txt:
            (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        dataset = LoadImages(im_path, img_size=self.imgsz, stride=self.stride)

        t0 = time.time()
        # event fire
        pevent.enterjob(im_path)
        for path, img, im0s, vid_cap in dataset:
            detail_js = {"fn": path.replace(image_pt_path, ""), "ng": -1, "boxs": []}

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h !=
                    img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=True)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=True)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, 0.8, 0.45, classes=[0,1], agnostic=True)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # print("im0.shape:", im0.shape)
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                save_path = str(save_dir)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if self.save_txt:  # Write to file
                            line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.view_img:  # Add bbox to image
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        # add result
                        detail_js["ng"] = int(cls)
                        zh, zw, zc = im0.shape
                        xywh_world = [round(xywh[0]*zw), round(xywh[1]*zh), round(xywh[2]*zw), round(xywh[3]*zh)]
                        detail_js["boxs"].append({"ng": int(cls), "conf": round(float(conf), 3), "xywh": xywh_world})

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
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
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
            # end of one img
            if len(detail_js["boxs"]):
                box_clss = [0, 0] # 2 class
                for box in detail_js["boxs"]:
                    box_clss[box["ng"]] += 1
                detail_js["ng"] = 0 if box_clss[0] > box_clss[1] else 1


            if self.debug_print:
                print("detail_js:", detail_js)
            ret_json["detail"].append(detail_js)
        # /for loop end 
        if self.save_txt or self.save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        # event fire
        pevent.finishjob('./'+str(save_dir))

        # if can we delete input-dir
        # shutil.rmtree(im_path)

        ngsum = [0, 0] # only 2 class
        for dt in ret_json["detail"]:
            if dt["ng"] > -1:
                ngsum[dt["ng"]] += 1

        # print("ngsum:", ngsum)
        if ngsum[0]+ngsum[1] > 0:
            ret_json["ng"] = 0 if ngsum[0] > ngsum[1] else 1
        else:
            ret_json["ng"] = 0 # if has no ng or good

        # return 200 if ret_json["ng"]==0 else 400, ret_json
        return 200, ret_json





#
# run :   DEBUG_PRINT=false python detect_web.py 
#
if __name__ == '__main__':
    print('*** HV DETECTOR SINGLE RUN ***')

    save_img = os.environ.get('SAVE_IMG', "false").lower() == "true"
    debug_print = os.environ.get('DEBUG_PRINT', "true").lower() == "true"
    image_path = os.environ.get('IMAGE_PATH', './inference/images')
    web_debug = os.environ.get('WEB_DEBUG', "false").lower() == "true"

    print('==========MAIN-CONFIG===========')
    print(f" - save_img:{save_img}")
    print(f" - debug_print:{debug_print}")
    print(f" - image_path:{image_path}")
    print(f" - web_debug:{web_debug}")
    print("==========================")

    # get detector
    detector = HvDetect(save_img, debug_print, image_path, web_debug)
    if len(sys.argv)>1:
        req_id = sys.argv[1]
        code, res_js = detector.run_detect(req_id)
        print("----")
        print(code)
        print(json.dumps(res_js, indent=4))
        print("----")



