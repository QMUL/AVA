import sys, os, cv2, time
import numpy as np, math
from argparse import ArgumentParser
#try:
#    from armv7l.openvino.inference_engine import IECore, IENetwork, IEPlugin
try:
	from openvino.inference_engine import IECore, IENetwork, IEPlugin
except:
	print('OpenVINO not installed')


yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 80
coords = 4
num = 3
anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]


label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


class YOLOv3ov(object):
    def __init__(self):
        self.img_w = 1920
        self.img_h = 1080
        self.m_input_size = 416

        self.new_w = int(self.img_w * self.m_input_size/self.img_w)
        self.new_h = int(self.img_h * self.m_input_size/self.img_h)

        model_xml = "./detector/YOLOv3_ov/lrmodels/YoloV3/FP32/frozen_yolo_v3.xml" #<--- CPU
        #model_xml = "lrmodels/YoloV3/FP16/frozen_yolo_v3.xml" #<--- MYRIAD
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        '''
        # Older version
        plugin = IEPlugin(device='CPU')
        plugin.add_cpu_extension("./detector/YOLOv3_ov/lib/libcpu_extension.so")
        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.exec_net = plugin.load(network=net)
        '''

        ieCore = IECore()
        #net = ieCore.read_network(model=model_xml, weights=model_bin)
        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.exec_net = ieCore.load_network(network=net, device_name="CPU")

    def __call__(self, image):

        resized_image = cv2.resize(image, (self.new_w, self.new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((self.m_input_size, self.m_input_size, 3), 128)
        canvas[(self.m_input_size-self.new_h)//2:(self.m_input_size-self.new_h)//2 + self.new_h,(self.m_input_size-self.new_w)//2:(self.m_input_size-self.new_w)//2 + self.new_w,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        outputs = self.exec_net.infer(inputs={self.input_blob: prepimg})

        objects = []

        for output in outputs.values():
            objects = self.ParseYOLOV3Output(output, self.new_h, self.new_w, self.img_h, self.img_w, 0.7, objects)

        # Filtering overlapping boxes
        objlen = len(objects)
        for i in range(objlen):
            if (objects[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (self.IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                    objects[j].confidence = 0
        
        # Adapt the outputs 
        #bbox_xywh (N,4)
        #cls_conf (N,)
        #cls_ids (N,)
        bbox     = []
        conf     = []
        class_id = []
        for obj in objects:
            x0 = obj.xmin 
            y0 = obj.ymin
            x1 = obj.xmax 
            y1 = obj.ymax
            w=x1-x0
            h=y1-y0
            if w>0 and h>0:
                bbox.append([x0+(w/2.),y0+(h/2.),w,h])
                conf.append(obj.confidence)
                class_id.append(obj.class_id)

        bbox = np.array(bbox).reshape(-1,4).astype(float)
        conf = np.array(conf).reshape(-1)
        class_id = np.array(class_id).reshape(-1).astype(int)

        if bbox.shape[0]:
            return bbox, conf, class_id
        else:
            return None,None,None
    

    def EntryIndex(self, side, lcoords, lclasses, location, entry):
        n = int(location / (side * side))
        loc = location % (side * side)
        return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)    


    def IntersectionOverUnion(self, box_1, box_2):
        width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
        height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
        area_of_overlap = 0.0
        if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
            area_of_overlap = 0.0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
        box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
        area_of_union = box_1_area + box_2_area - area_of_overlap
        retval = 0.0
        if area_of_union <= 0.0:
            retval = 0.0
        else:
            retval = (area_of_overlap / area_of_union)
        return retval


    def ParseYOLOV3Output(self, blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

        out_blob_h = blob.shape[2]
        out_blob_w = blob.shape[3]

        side = out_blob_h
        anchor_offset = 0

        if len(anchors) == 18:   ## YoloV3
            if side == yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == yolo_scale_52:
                anchor_offset = 2 * 0

        elif len(anchors) == 12: ## tiny-YoloV3
            if side == yolo_scale_13:
                anchor_offset = 2 * 3
            elif side == yolo_scale_26:
                anchor_offset = 2 * 0

        else:                    ## ???
            if side == yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == yolo_scale_52:
                anchor_offset = 2 * 0

        side_square = side * side
        output_blob = blob.flatten()

        for i in range(side_square):
            row = int(i / side)
            col = int(i % side)
            for n in range(num):
                obj_index = self.EntryIndex(side, coords, classes, n * side * side + i, coords)
                box_index = self.EntryIndex(side, coords, classes, n * side * side + i, 0)
                scale = output_blob[obj_index]
                if (scale < threshold):
                    continue
                x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
                y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
                height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
                width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
                for j in range(classes):
                    class_index = self.EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)               
                    prob = scale * output_blob[class_index]
                    if prob < threshold:
                        continue
                    obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                    if obj.class_id==0:
                        objects.append(obj)
        return objects
