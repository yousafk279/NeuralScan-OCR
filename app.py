
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import json
import gradio as gr

# Model Architecture
class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(OCRModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.lstm1 = nn.LSTM(512, hidden_size, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 99)

    def forward(self, x):
        conv = self.cnn(x).squeeze(2).permute(2, 0, 1)
        output, _ = self.lstm1(conv)
        T, B, H = output.size()
        output = self.linear1(output.view(T * B, H)).view(T, B, -1)
        output, _ = self.lstm2(output)
        return self.fc(output.view(output.size(0) * output.size(1), -1)).view(output.size(0), output.size(1), -1)

# Load resources
device = torch.device('cpu')
with open('en_ocr_metadata.json', 'r') as f:
    meta = json.load(f)

model = OCRModel(99).to(device)
model.load_state_dict(torch.load('english_ocr_specialist_v1.pth', map_location=device))
model.eval()

def predict(image):
    if image is None: return "Please upload an image."
    img = image.convert('L')
    transform = T.Compose([T.Resize((32, 256)), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        indices = torch.argmax(logits, dim=2)[:, 0].tolist()
        res = []
        for i in range(len(indices)):
            if indices[i] != 0 and (i == 0 or indices[i] != indices[i-1]):
                res.append(meta['num_to_char'][str(indices[i])])
        return "".join(res)

# Professional UI
theme = gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")
with gr.Blocks(theme=theme, title="Neural OCR") as demo:
    gr.Markdown("# ⚡ NeuralScan: High-Precision OCR\n*CRNN + Bi-LSTM Architecture*")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload Text Image")
            btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            text_out = gr.Textbox(label="Recognized Text")
    btn.click(predict, img_in, text_out)

demo.launch()
