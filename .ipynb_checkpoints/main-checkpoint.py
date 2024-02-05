import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from bisenetv2 import BiSeNetV2
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

modelweight = "model_segmentation_realtime_skin_30.pth"
state_dict = torch.load(modelweight, map_location=torch.device('cpu'))
model = BiSeNetV2(['skin'])
model.load_state_dict(state_dict)

model.eval()

def createSkinMask(targetimage):
    targetimage = cv2.cvtColor(targetimage,cv2.COLOR_BGR2RGB)

    image_width = (targetimage.shape[1] // 32) * 32
    image_height = (targetimage.shape[0] // 32) * 32

    resized_image = targetimage[:image_height, :image_width]

    fn_image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])

    transformed_image = fn_image_transform(resized_image)

    with torch.no_grad():
        transformed_image = transformed_image.unsqueeze(0)
        results = model(transformed_image)['out']
        results = torch.sigmoid(results)

        results = results > 0.5
        mask = results[0]
        mask = mask.squeeze(0)
        mask = mask.cpu().numpy()
        mask = mask * 255
        mask = mask.astype('uint8')

    return mask,resized_image

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')

    original = cv2.imread("henderson1.png")
    # cv2.imshow("raw Image",original)
    print("reading image")
    mask, resized_image = createSkinMask(original)

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    ontop = cv2.addWeighted(mask, 0.5, gray, 0.5, 0)

    f, axarr = plt.subplots(1, 3, figsize=(20, 15))
    axarr[0].imshow(mask, cmap='gray')
    axarr[1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axarr[2].imshow(ontop, cmap='gray')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
