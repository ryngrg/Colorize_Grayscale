import torchvision.transforms
import torchvision.transforms as T
from train import Trainer
import torch
from PIL import Image


def infer(model_name, image_path):
    input_transform = T.Compose([T.ToTensor(),
                                 T.Resize(size=(256, 256)),
                                 T.Grayscale(),
                                 T.Normalize((0.5), (0.5))
                                 ])
    image = Image.open(image_path, mode="r")
    input_image = input_transform(image)
    model = torch.load(r"../models/" + model_name)
    model.eval()
    output = model(input_image)
    output_transform = torchvision.transforms.ToPILImage()
    output_image = output_transform(output)
    output_image.show()


if __name__ == "__main__":
    trainer = Trainer()

    # set trying to False to get final model trained on all data
    trying = True

    if trying:
        trainer.train(["train_files.csv"], "temp_model")
        print("Validation loss:", trainer.validate())
    else:
        trainer.train(["train_files.csv", "val_files.csv"], "final_model")

        # put image path for image you want to test the final model on
        image_path = r"../data/1000.jpg"

        infer("final_model", image_path)
