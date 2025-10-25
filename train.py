from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'./yolo11-pose.yaml')
    # model.load('yolov11m.pt') # 是否加载预训练权重
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=100,
                single_cls=True,  # 多类别设置False
                batch=16,
                workers=10,
                device='0',
                )
