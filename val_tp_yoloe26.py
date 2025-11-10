import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from ultralytics import YOLOE

# Create a YOLOE model

model_path="/root/ultra_louis_work/runs/yoloe26s_tp/mobileclip2:b_26s_bs128_ptwobject365v1_cls5_enginedata_exp2/weights/best.pt"

# model = YOLOE("/home/laughing/Downloads/best_yoloe26s.pt")
model = YOLOE(model_path)

# del model.model.model[-1].one2one_cv2
# del model.model.model[-1].one2one_cv3
# model.model.end2end = False

# Conduct model validation on the COCO128-seg example dataset
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 catIds=all] = 0.294
metrics = model.val(data="lvis.yaml", split="minival", max_det=1000,conf=0.0001)