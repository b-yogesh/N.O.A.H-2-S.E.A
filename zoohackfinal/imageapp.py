import torchvision.transforms as transforms
import os
import torch
from PIL import Image

model_file = "/Users/rajaathota72/PycharmProjects/zoohackfinal/trainedModel.pth"
trainedModel = torch.load(model_file, map_location=torch.device('cpu'))

def run_prediction(input):
  input = preprocess(input).unsqueeze_(0)
  trainedModel.eval()
  input = input
  outputs = trainedModel(input)
  _, preds = torch.max(outputs, 1)
  p = preds.cpu().numpy()
  return p

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


test_img = Image.open('/Users/rajaathota72/PycharmProjects/zoohackfinal/test/004.Groove_billed_Ani/Groove_Billed_Ani_0005_1750.jpg')
pred = run_prediction(test_img)
print(pred)

def image(input):
    test_img = Image.open(input)
    pred = run_prediction(test_img)
    return pred
def main():
    import streamlit as st
    st.title("Predict the bird_spieces - Test Program")
    import io
    file_buffer = st.file_uploader("Upload the image")
    text_io = io.TextIOWrapper(file_buffer)
    print(text_io)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if st.button("Recognise"):
        result = image(file_buffer)
        if result > 0 and result < 30:
            st.success("Groove billed is the bird species identified")
            st.image(file_buffer, width = 650)
            st.write("This is an endangered species and Please report to help")
            st.button("report now Anonymously")
        elif result >30 and result < 60:
            st.success("Another bird")
            st.write("This is an endangered species. Dont use this products")
if __name__ == "__main__":
    main()