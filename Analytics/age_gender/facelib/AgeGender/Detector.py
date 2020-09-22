from AgeGender.models.model import ShuffleneTiny, ShuffleneFull
import torch

from inference import load_to_IE, sync_inference, async_inference, get_async_output

class AgeGender:

    def __init__(self, name, weight_path, device, ov):
        """
        Age and gender Detector
        :param name: name of backbone (full or tiny)
        :param device: model run in cpu or gpu (cuda, cpu)
        :param weight_path: path of network weight

        Notice: image size must be 112x112
        but cun run with 224x224

        Method detect:
                :param faces: 4d tensor of face for example size(1, 3, 112, 112)
                :returns genders list and ages list
        """
        if name == 'tiny':
            model = ShuffleneTiny()
        elif name == 'full':
            model = ShuffleneFull()
        else:
            exit('from AgeGender Detector: model dose not support just(tiny, full)')

        if not ov:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
            model.to(device).eval()
        else:
            model = load_to_IE(weight_path.replace('.pth','.xml'))

        self.model = model
        self.device = device
        self.ov = ov

    def detect(self, faces):
        faces = faces.permute(0, 3, 1, 2)
        faces = faces.float().div(255).to(self.device)

        mu = torch.as_tensor([0.485, 0.456, 0.406], dtype=faces.dtype, device=faces.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=faces.dtype, device=faces.device)
        faces[:].sub_(mu[:, None, None]).div_(std[:, None, None])

        if not self.ov:
            outputs = self.model(faces)
        else:
            outputs = sync_inference(self.model, image=faces.numpy())
            outputs = torch.from_numpy(outputs['scores'])

        genders = []
        ages = []
        for out in outputs:
            gender = torch.argmax(out[:2])
            #gender = 'Male' if gender == 0 else 'Female'
            genders.append(gender)
            ages.append(int(out[-1]))

        return genders, ages
