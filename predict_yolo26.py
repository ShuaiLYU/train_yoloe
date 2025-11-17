

import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from ultralytics import YOLO

# Initialize a YOLOE model
model_path="./yolo26s-objv1.pt"

model=YOLO(model_path)  # or select yoloe-11s/m-seg.pt for different sizes



im_file="ultralytics/assets/bus.jpg"
# Run detection on the given image
results = model.predict(im_file,conf=0.1,iou=0.99)
# save results
results[0].save("../runs/predict_yolo26.jpg")


# check the names of results
print("results names:",results[0].names)
