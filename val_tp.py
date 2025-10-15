import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE

# Create a YOLOE model

model=YOLOE("yoloe-11l.yaml").load("yoloe-11l-seg.pt")
# model=YOLOE("/root/ultra_louis_work/runs/yoloe_origina/l_train_tp/11s_0.002_close2_ep30_exp13/weights/best.pt")

# Conduct model validation on the COCO128-seg example dataset
metrics = model.val(data="../datasets/lvis.yaml",split="minival",max_det=1000)