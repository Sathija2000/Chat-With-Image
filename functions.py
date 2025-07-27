from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

def get_image_caption(image_path):
    """
    Generate a caption for the given image using a pre-trained BLIP model.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        str: The generated caption for the image.
    """

    # Open the image
    image = Image.open(image_path).convert("RGB")

    model_name = "Salesforce/blip-image-captioning-base"
    device = "cpu"

    # Load the BLIP processor and model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Process the image and generate the caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)

    # Decode the generated caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


def get_object_detection(image_path):
    """
    Perform object detection on the given image using a pre-trained DETR model.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        list: A list of detected objects with their labels and bounding boxes.
    """

    # Open the image
    image = Image.open(image_path).convert("RGB")

    model_name = "facebook/detr-resnet-50"
    device = "cpu"

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detected_objects += '[{}, {}, {}, {}]'.format(
            int(box[0]), int(box[1]), int(box[2]), int(box[3])
        )
        detected_objects += ' {}'.format(model.config.id2label[int(label)])
        detected_objects += ' {}\n'.format(float(score))

    return detected_objects

if __name__ == "__main__":
    # Example usage
    image_path = "images/image_3.jpeg"
    detections = get_object_detection(image_path)
    print(f"Generated Caption: {detections}")
