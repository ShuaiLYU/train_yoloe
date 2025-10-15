

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch



model_version="11s"
model = YOLOE("yoloe-{}.yaml".format(model_version)).load("yoloe-{}-seg.pt".format(model_version))

print(model.model._clip_weight)  # should print "mobileclip:blt"