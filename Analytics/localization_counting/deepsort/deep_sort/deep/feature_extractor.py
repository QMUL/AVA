import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .model import Net

from .inference import load_to_IE, sync_inference, async_inference, get_async_output

class Extractor(object):
    def __init__(self, model_path, use_cuda=True, ov=False):
        self.ov = ov
        if not self.ov:
            self.net = Net(reid=True)
            self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
            self.net.load_state_dict(state_dict)
            print("Loading weights from {}... Done!".format(model_path))
            self.net.to(self.device)
        else:
            self.net = load_to_IE(model_path.replace('t7','xml'))
            self.device = torch.device("cpu")
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        if self.ov:
            im_batch =im_batch.numpy()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
         
        if not self.ov:
            with torch.no_grad():
                im_batch = im_batch.to(self.device)
                features = self.net(im_batch).cpu().numpy()
        else:
            features=[]
            for im in im_batch:
                features.append(sync_inference(self.net, image=im)['Div_75/mul_'])
            features = np.stack(features).reshape(-1,512)
        return features


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

