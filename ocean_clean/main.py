
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image



def main():
    print("Hello ocean clean")
    classify_image('data/picture.png')
    classify_image('data/picture2.png')

def classify_image(path):

    # get image from camera
    image = get_image_from_camera(path)

    # classify imagine using CLIP
    label = classify_image_using_clip(image)

    # display image with label
    display_image(path, image, label)

def get_image_from_camera(path):
    return Image.open(path)


def classify_image_using_clip(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    processed_image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(["plastic", "people"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(processed_image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(processed_image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("probs", probs)

    label = "plastic" if probs[0][0] > probs[0][1] else "not plastic"

    return label

def display_image(path, image, label):
    imgplot = plt.imshow(image)
    plt.xlabel(label) 
    plt.savefig(path + ".result.png")

     

main()
