import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import deque

class MotionDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torchvision.models.video.r3d_18(pretrained=True).to(device)
        self.model.eval()

        # 16 frame'lik pencere (modelin input uzunluğu)
        self.frames = deque(maxlen=16)

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])
        ])

    def predict(self, frame):
        # Frame ekle
        self.frames.append(frame)

        # Yeterli frame yoksa bekle
        if len(self.frames) < 16:
            return "Bekleniyor..."

        # Frame'leri tensora dönüştür
        clip = [self.transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in list(self.frames)]
        clip = torch.stack(clip).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        # Model tahmini
        with torch.no_grad():
            preds = self.model(clip)
            pred_score = torch.nn.functional.softmax(preds, dim=1)
            max_score = torch.max(pred_score).item()

        # Basit eşikleme: eğer modelin eminliği yüksekse "ani hareket" say
        if max_score > 0.8:
            return "Ani Hareket"
        else:
            return "Normal"
