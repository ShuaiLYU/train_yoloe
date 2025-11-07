import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator

def read_pf_det_from_seg_fused(model_path,yaml_name):

    # load the most model weights
    det_model = YOLOE(yaml_name).load(model_path)
    import copy
    seg_model=YOLOE(model_path)

    # copy the lrpc model ()
    det_model.model.model[-1].lrpc =copy.deepcopy(seg_model.model.model[-1].lrpc)
    det_model.model.model[-1].is_fused = True
    det_model.model.model[-1].conf = 0.001
    det_model.model.model[-1].max_det = 1000

    # del the last layer of loc and cls head (which is copied from the set_vocab function)
    import torch.nn as nn
    for loc_head, cls_head in zip(det_model.model.model[-1].cv2, det_model.model.model[-1].cv3):
        assert isinstance(loc_head, nn.Sequential)
        assert isinstance(cls_head, nn.Sequential)
        del loc_head[-1]
        del cls_head[-1]

    det_model.model.names=seg_model.model.names

    return det_model





def read_pf_det_from_seg_unfused(model_path,yaml_name,unfused_model_weight):

    # load the most model weights
    det_model = YOLOE(yaml_name).load(model_path)


    # set vocab from the unfused model
    unfused_model=YOLOE(unfused_model_weight)
    unfused_model.eval()
    unfused_model.cuda()

    with open('../datasets/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]
    vocab = unfused_model.get_vocab(names)


    det_model.eval()
    det_model.set_vocab(vocab, names=names)
    det_model.model.model[-1].is_fused = True
    det_model.model.model[-1].conf = 0.001
    det_model.model.model[-1].max_det = 1000

    return det_model




model_weight=os.path.abspath("./yoloe-11s-seg-pf.pt")
model=read_pf_det_from_seg_fused(model_weight,"yoloe-11s.yaml")

# val: Scanning /root/autodl-tmp/datasets/lvis/labels/val2017.updated.cache... 4752 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4752/4752 24.7Mit/s 0.0s
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 297/297 10.7it/s 27.6s
#                    all       4752      50672      0.288      0.243       0.14     0.0906
#        3D CG rendering       4752      50672      0.288      0.243       0.14     0.0906



# model_weight="/home/louis/ultra_louis_work/ultralytics/runs/detect/train14/weights/best.pt"
model_weight="/root/ultra_louis_work/ultralytics/runs/yoloe_11s_pf/epo5_bs129_lr0.002_maxdet1000_closemosaic2_singleclsTrue2/weights/last.pt"
version="11s"
import sys
if len(sys.argv)>3:
    model_weight=sys.argv[1]
    version=sys.argv[2]
model=read_pf_det_from_seg_unfused(model_weight,f"yoloe-{version}.yaml",f"yoloe-{version}-seg.pt")






# Conduct model validation on the COCO128-seg example dataset
metrics = model.val(data="lvis.yaml",split="minival", single_cls=True ,max_det=1000) # map 0


# val: Scanning /root/autodl-tmp/datasets/lvis/labels/val2017.updated.cache... 4752 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4752/4752 65.8Mit/s 0.0s
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 297/297 7.9it/s 37.5s
#                    all       4752      50672      0.297      0.315      0.189      0.119
#        3D CG rendering       4752      50672      0.297      0.315      0.189      0.119
# Speed: 0.0ms preprocess, 2.2ms inference, 0.0ms loss, 2.4ms postprocess per image
