from ultralytics import YOLO
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*adaptive_max_pool2d_backward_cuda does not have a deterministic implementation.*"
)

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26/yolo26n-C3k2_FCM.yaml')  # build from YAML and transfer weights
    # model = YOLO('yolo26n.pt')   
        # Train the model
    model.train(data='ultralytics/cfg/datasets/data_origin.yaml',
                pretrained='yolo26n.pt', 
                epochs=140,
                imgsz=960,
                batch=16,
                device=0,
                workers=10,
                project='runs_ship',
                name='26n_c3k2_fcm',
                )
