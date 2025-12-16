import torch
import torchvision
import cv2
import torchvision.transforms as transforms

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from threading import Thread
from time import sleep

# this function is used as a thread
def create_window(c2):
    # load a pre-trained model
    print("Load model")

    model = fasterrcnn_resnet50_fpn\
            (
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )

    model.eval()

    print("Done")
    print("")

    print("Initialize camera")

    # initialize user camera
    cap = c2.VideoCapture(0)

    print("Done")
    print("")

    # define the transformation
    transform = transforms.Compose\
    ([
        transforms.ToTensor()
    ])

    ret, frame = cap.read()

    if not ret:
        return

    # convert the image to a PyTorch tensor
    image = c2.cvtColor(frame, c2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0)

    # run the model
    print("Make predictions")

    with torch.no_grad():
        predictions = model(image_tensor)

    print("Done")
    print("")

    # process predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    print("Identify objects")

    count = 0

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            count += 1
            print(f"Object {count} identified")

            x1, y1, x2, y2 = box.int().tolist()

            c2.rectangle\
            (
                frame, 
                (x1, y1), 
                (x2, y2), 
                (0, 255, 0), 
                2
            )

    if count == 0:
        print("No objects identified")

    print("Done")    
    print("")

    print("Show window")

    c2.startWindowThread()

    c2.namedWindow("Camera", c2.WINDOW_AUTOSIZE)        
    c2.imshow('Camera', frame)

    print("Done")
    print("")

    # exit camera 
    print("Exit camera")

    cap.release()

    print("Done")
    print("")

    if cv2.waitKey(0) == ord('q'):
        print("Exit window")

        if cap.isOpened():
            cap.release()

        c2.destroyAllWindows()

        print("Done")
        print("")        
    else:
        print("Pause")

        sleep(1.5) # make a pause 

        print("Done")
        print("")

        print("Exit window")

        if cap.isOpened():
            cap.release()

        c2.destroyAllWindows()

        print("Done")
        print("")

if __name__ == "__main__":    
    print("")

    while True:
        print("Create Thread")

        thread = Thread\
                (
                    target = create_window, 
                    args = (cv2, )
                )

        print("Done")
        print("")

        thread.start()
        thread.join()

        print("Thread terminated")
        print("")

        response = input("Do you want to continue? (Y/N): ")
        
        print("")

        if response.lower().strip() == "n":
            break

    print("")
