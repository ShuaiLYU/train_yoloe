
import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch







data = dict(
    train=dict(
        yolo_data=["coco128.yaml"],
        grounding_data=[
            # dict(
            #     img_path="../datasets/flickr/full_images/",
            #     json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
            # ),
            # dict(
            #     img_path="../datasets/mixed_grounding/gqa/images"
            # ,
            #     json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            # ),
        ],
    ),
    val=dict(yolo_data=["../datasets/lvis.yaml"]),
)




# model = YOLOE("/root/ultra_louis_work/yoloe/yoloe-v8s-seg-det.pt")

import argparse

parser=argparse.ArgumentParser(description="train yoloe with visual prompt")
parser.add_argument("--model_version", type=str, default="v8s")
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--close_mosaic", type=int, default=0)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--project", type=str, default="runs/quick_verify")
#device
parser.add_argument("--device", type=str, default="0")
# val
parser.add_argument("--val", type=bool, default=True)
parser.add_argument("--name", type=str, default="yoloe_vp")
parser.add_argument("--clip_weight_name", type=str, default="mobileclip2:b")  # mobileclip2b
#
args = parser.parse_args()



model = YOLOE("yoloe-{}.yaml".format(args.model_version))


model.train(
    data=data,
    batch=args.batch,
    epochs=args.epochs,
    close_mosaic=args.close_mosaic,
    optimizer="AdamW",
    lr0=args.lr, # for s/m, please set lr0=8e-3
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=4,
    max_det=1000,
    trainer=YOLOETrainerFromScratch,  # use YOLOEVPTrainer if converted to detection model
    clip_weight_name=args.clip_weight_name,
    device=args.device,
    save_period=5,
    val=args.val,
    project=args.project,
    name=args.name,
)
