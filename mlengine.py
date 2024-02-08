from email.mime import image
import io
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Define the neural network class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)  # Flattening the input
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Create an instance of the neural network
input_size = 784
hidden_size = 500
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

# Load the trained model weights from a file
MODEL_PATH = "mnist_ffn.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Define a function to transform the input image
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# Define a function to get the model prediction
def get_prediction(image_tensor):
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted

# Example usage
# image = load_and_preprocess_image('example_image.jpg')
# prediction = get_prediction(image)
# print(prediction)
