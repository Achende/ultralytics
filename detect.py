import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/pose/train3/weights/best.pt')
    model.predict(source='data/enginer_image/blue_232.jpg',
                  imgsz=640,
                  device='0',
                  save=True
                  )
