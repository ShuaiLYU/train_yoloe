import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)


from ultralytics import YOLOE

# Initialize a YOLOE model
model_path="./yoloe-11s-seg.pt"
model=YOLOE("yoloe-11s-seg.yaml").load(model_path)

# model_path="/root/ultra_louis_work/runs/yoloe_original_train_tp/mobileclip:blt_11s_0.002_close2_ep30_prefix_exp2/weights/best.pt"
# model = YOLOE(model_path)  # or select yoloe-11s/m-seg.pt for different sizes



# update the args clip_weight_name 
model.args["clip_weight_name"]="mobileclip:blt"
# Set text prompt to detect person and bus. You only need to do this once after you load the model.



names = ["a person", "bus","a bus","people","a people","two people","a group of people","three people"]

names=["person","people"]

model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("ultralytics/assets/bus.jpg",conf=0.1,iou=0.99)

# Show results
results[0].save("predicted_image.jpg")