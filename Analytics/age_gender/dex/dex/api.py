import os
import cv2
import torch
import numpy as np

from .models import Age, Gender

from inference import load_to_IE, sync_inference, async_inference, get_async_output


class api:
    def __init__(self, device, ov):
        cwd = os.path.dirname(__file__)
        age_model_path = os.path.join(cwd, 'pth/dex_age.pth')
        gender_model_path = os.path.join(cwd, 'pth/dex_gender.pth')

        if not ov:
            self.age_model = Age()
            self.age_model.load_state_dict(torch.load(age_model_path))
            self.age_model.eval()
            self.age_model.to(device)
            
            self.gender_model = Gender()
            self.gender_model.load_state_dict(torch.load(gender_model_path))
            self.gender_model.eval()
            self.gender_model.to(device)
        else:
            self.age_model = load_to_IE(age_model_path.replace('.pth','.xml'))
            self.gender_model = load_to_IE(gender_model_path.replace('.pth','.xml'))

        self.device = device
        self.ov = ov

    def expected_age(self, vector):
        ages=[]
        for i in range(len(vector)):
            res = [(i+1)*v for i, v in enumerate(vector[i])]
            ages.append(int(round(sum(res))))
        return ages

    def expected_gender(self, vector):    # model outputs, 0:female, 1:male. We need the opostte
        genders=[]
        for i in range(len(vector)):
            genders.append(0 if vector[i][1]>vector[i][0] else 1)
        return genders

    def estimate(self, img):

        if not self.ov:
            with torch.no_grad():
                img = img.to(self.device)
                genders = self.expected_gender(self.gender_model(img).detach().cpu().numpy()) 
                ages = self.expected_age(self.age_model(img).detach().cpu().numpy())
        else:
            genders = sync_inference(self.gender_model, image=img.numpy())
            genders = self.expected_gender(genders['scores'])

            ages = sync_inference(self.age_model, image=img.numpy())
            ages = self.expected_age(ages['scores'])
            
        return ages, genders
        