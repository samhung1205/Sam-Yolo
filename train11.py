from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n-asff.yaml')  # build from YAML and transfer weights
    # model = YOLO('yolo11n.pt')    
        # Train the model
    model.train(data='ultralytics/cfg/datasets/data_merge.yaml',
                pretrained='yolo11n.pt', 
                epochs=150,
                imgsz=960,
                batch=16,
                device=0,
                workers=10,
                project='runs_merge',
                name='11n_asff',
                )
