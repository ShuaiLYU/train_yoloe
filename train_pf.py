
import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPEFreeTrainer


DATA_DIR="../datasets/"

Objects365v1="../datasets/Objects365v1.yaml"

data = dict(
    train=dict(
        yolo_data=[Objects365v1],
        grounding_data=[
            dict(
                img_path=DATA_DIR+"flickr/full_images/",
                json_file=DATA_DIR+"flickr/annotations/final_flickr_separateGT_train_segm.json",
            ),
            dict(
                img_path=DATA_DIR+"mixed_grounding/gqa/images",
                json_file=DATA_DIR+"mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            ),
        ],
    ),
    val=dict(yolo_data=["./lvis.yaml"]),
)



# model = YOLOE("/root/ultra_louis_work/yoloe/yoloe-v8s-seg-det.pt")
model = YOLOE("yoloe-11s.yaml").load("yoloe-11s-seg.pt")

# reinit the model.model.savpe.
# model.model.model[-1].savpe.init_weights() 
# assert False, "check the weights have been reinit"



# freeze layers.
head_index = len(model.model.model) - 1
freeze = [str(f) for f in range(0, head_index)]
for name, child in model.model.model[-1].named_children():
    if "cv3" not in name:
        freeze.append(f"{head_index}.{name}")

freeze.extend(
    [
        f"{head_index}.cv3.0.0",
        f"{head_index}.cv3.0.1",
        f"{head_index}.cv3.1.0",
        f"{head_index}.cv3.1.1",
        f"{head_index}.cv3.2.0",
        f"{head_index}.cv3.2.1",
    ]
)


project="yoloe_11s_pf"
epochs=5
batch=129
device="0,1,2"
max_det=1000
lr= 2e-3
close_mosaic=2
single_cls=True  # this is needed

name=f"epo{epochs}_bs{batch}_lr{lr}_maxdet{max_det}_closemosaic{close_mosaic}_singlecls{single_cls}"





model.train(
    data=data,
    batch=batch,
    epochs=epochs,
    close_mosaic=close_mosaic,
    optimizer="AdamW",
    lr0=lr,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    trainer=YOLOEPEFreeTrainer,
    device=device,
    freeze=freeze,
    single_cls=single_cls,
)