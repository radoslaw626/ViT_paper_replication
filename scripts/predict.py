import torch
import torchvision
import argparse
import model_builder


"""
Performs image classification prediction using a pre-trained PyTorch model.

This script loads a pre-trained TinyVGG model and performs a prediction on a specified image. The user can provide the image filepath and the model filepath as command-line arguments. The script is designed to work with a model trained to classify images into three categories: pizza, steak, and sushi.

Usage:
    python scripts/predict.py --image "path/to/image.jpg" --model_path "models/model_file.pt"

Arguments:
    --image (str): Filepath to the target image to predict on.
    --model_path (str): Filepath to the target model to use for prediction. Default is 'models/simple_data_tinyvgg_model.pt'.

The script will output the predicted class and the associated probability for the given image.

Example:
    python scripts/predict.py --image "data/test/pizza.jpg" --model_path "models/simple_data_tinyvgg_model.pt"
"""

parser = argparse.ArgumentParser()

parser.add_argument("--image",
                    help="target image filepath to predict on")

parser.add_argument("--model_path",
                    default="models/simple_data_tinyvgg_model.pt",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

class_names = ["pizza", "steak", "sushi"]

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_PATH = args.image
print(f"[INFO] Predicting on {IMG_PATH}")

def load_model(filepath=args.model_path):
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=10,
                                output_shape=3).to(device)

  print(f"[INFO] Loading in model from: {filepath}")                              
  model.load_state_dict(torch.load(filepath))

  return model

def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
  model = load_model(filepath)

  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  image = image / 255.

  transform = torchvision.transforms.Resize(size=(64, 64))
  image = transform(image) 

  model.eval()
  with torch.inference_mode():
    image = image.to(device)

    pred_logits = model(image.unsqueeze(dim=0))

    pred_prob = torch.softmax(pred_logits, dim=1)

    pred_label = torch.argmax(pred_prob, dim=1)
    pred_label_class = class_names[pred_label]

  print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  predict_on_image()
