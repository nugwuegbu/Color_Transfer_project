import cv2
import torch
import imutils
from torchvision import transforms
import matplotlib.pyplot as plt
from bisenetv2 import BiSeNetV2
import random,glob
import numpy as np
import pandas as pd
from color_transfer import *
from  color_detection import extract_dominant_color,createSkin


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









# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = imutils.url_to_image("https://raw.githubusercontent.com/octalpixel/Skin-Extraction-from-Image-and-Finding-Dominant-Color/master/82764696-open-palm-hand-gesture-of-male-hand_image_from_123rf.com.jpg")
    target_image = imutils.url_to_image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAMAAzAMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAACAwEEBQYAB//EAEEQAAEDAgQCBgcECAYDAAAAAAEAAgMEEQUSITFBUQYTImFxkRQyQoGhsdEjUmLBFSQzcoKS4fAHQ1ODssIWNET/xAAYAQADAQEAAAAAAAAAAAAAAAAAAQIDBP/EACERAQEAAgICAwADAAAAAAAAAAABAhEDIRIxE0FRBCJx/9oADAMBAAIRAxEAPwDqAFNl6ykBREPAKQFNlICYespspUgJgNlNkQC9ZADZesisvWQIGy9ZFZeskYLKLL0r442F0zsred7XS4p2y+w8DmeKD8b7giEJCaQhshJZCAhNIQEIEKIQOCcQgcEUyHBLcE9wSikZDgluCe5KcEgrPCUQrDwlEaoN0wCkKQpAVIQAiU2UgICLKQpspsmEWU2UhSgBsosjsvWQAWVetrIKOLPNJYn1Wgak9yqYzjUOHAxx5Zaq2jBs3vd9FzdO+atqXSzvL3n2jw7gpuX034+G5d1rQyzYhVh84ytB7Db6NWzbIzLy1VSkiZFEEyebgOYCuTUdNknpcbqxp5hRZGBlaByFlBCTgvsshCQmEISEEUQluTnBLISpwlwS3BPcEpwSBLglOCe4JTgkau8JRGqsPCURqg3SKV5SFaHgiUKQgJUheClAeAUoXSCJjnvcGMaLlxWNUdIYzdlEzPwzu2HuRvRzG5em04tYC+QgNA1J2XOYxj7w10OGb27Ux2931UObUYgCaiUvG4aNvJUJ6cMuALW4KbLXRhw6vbJhjcJczi5znG5c43JK2KeLq7SN0596GnprnM7ZXRCLWvpa6UkjqgoZ3OvYWV+gj614e72NzzPAKpRUVRVTA2McHF59rwW/HE2OIMY3K0fFVGHLyydRB11O6hMcgKbjAUJCYUBQCyEshOKW5AKcEtwTiluSpwhwSnBPcEpwSMh6SU96Sd0qbpApUBErQ8pCgIkBKkWuoWB0qxj0WIUNO79Yl1cR7Dfqlbo8Zus3pJjPpkvodM4dRGbOd98/RZ1ERnADrm+xXoqNrmXc06cbqxQ0kbZml1xfYqZjvt2Y9TTqaKMejtIHa4qtXUnXVHZ7JGpPAq7QR2YQHFHOyzSeQW+Xo8bdsRjDnMZ0stSgoBIOsmBDNw373j3IqCFk7Q9wBsdDz/vRamnAW5BYxHNy2dQDWgABuwGi8iXiFTk2EqFJUFAAQgITCgKQAUsppQOB5IBRCAo3EcwlvcwbuaPeg5KU4JbkbpYjtIPNLJB2I80U9Up4STunuSTuppuhUgrGd0jw7T7cEnbUWCaMepNjJG48csjTp4qtp1WqEXcsv9OU1r6W552/VU63pRR0sZcx4kdbRreP0RcoPG1oY3isOFUnWSWdK7SKO+rj9FwUU8lVUOnncXyvd2nuQVNXNidSZ53ZpL6C2jRyT6aJwc8WJ15LPvKurjw8fftr0g7JDhqrkUdix33ddlToi4uAc0i+moWiXdi7TsbFbz0pqUTrtNjrfdNrXjqnC1ieSpUb7WA2KdXO+ze4H2SAe/dFvRz2SMsFPAGSZS5xDDzP9hTLX1tO0ydWJ2gdqO13DvHMLNq5A+lo3m9mudqOBWhT17HxxufYOAOtuWhWc7aZYS902lxptS1pY1haRpYqw/ECz1ob+Dlzs1K2irM8V+qkcXN1sGk8Fcq6jMyMNIuBe548PNPHf2wy48fxqtxEO3hcP4lJr2fcd5rOjeDrfdWGljtLKtF8WKJ8XDOy2El34nKq7FZ36ANb3AIcViaIg9jdQdVkCotsLqb1V48WLWZU1Ep7Uzh4FVqh8sUo62VxbzJSGTk+qLKli75MmrzYi6Nxd459NIzsHrOui7M0Rbm13HiudZUEes4lW4MSjZxSmUPx0tiUNOgurUVS3uu5c4zFGCQnW19EbMQY6djWE5uV9U/OL1Pt172XjzNVd26GlkldTtD2uaBwIsUR3U1w8knl04FkLnO0HwT2NZE2xJunyFsYs0aqjUPN1hXXIKWUNZcbnTVZ4neZe0QBeyZUSE5R3KoDd57iiRNkb9A7M22xJ0K26O7ju3Wx3XN0L9TZ2vAFb9G5wPa00XRjS02aU6i41AV17QIQ0AOHJU6RwLlp5A+NuUa3utCBB2XAWsLclFc8ikkA8fgpdEWtJtezb7pU5zRlh52U096U4Xj0d8bxdosRfhzVbO6N7ra93cQmy2YTyP8AfzVbMTLZ4tb4qF7XoXF8bopjoD2b81m1lUX1TWRnss3F1anmEURdxA0vxWPG+73u4uPHdK0Y91u005b6xutKnk6w6aLBhf2T3bLQoZrEXTlVZprzRmSEtIGy4ivdJDUOaIJ3i+mSMuv5LtH1J6k5LukA0bwOi5STpJi7B1TpoacsFi2GAf8Ae6rNE39ApWYpO37HDKr/AHGGP/lZPm6O4tVsBnfBSM43eXnyGnxSBjNXJ+1rJ3dweW/KwT4XRzuzO7TuZN/jdROytz/BQdF6FhtVYs6Vw3bG9jB8bn4q9FgmBwa5Y3jm+cu+F7KnXNjZG1zbAhUTXt2sq8ZGdwyy+26KDBYjdlJQg/hjbdMEtHD6nVMvtZv0XP8ApYTpKpgpngjUC4KNQfF+1sOqoP8AUHxQGWPg8eRWBFXBvrNuntrWW2KrxlV8OP6ypNSSqlQ26uOe1Jlc0hc1Wy6gWd7lSa7LKb81o1ViT4LNGr9LJxNalI6xaQPet+keSLFYFFHcdh1j3rbp2yMtmFxbhqrl0eq6GiI58FqxOaIxqb7LDoHkAXBb46LYicco2WkyTYdI4CMC+nFVZT9nfvCOZ+nBVJnjqiNb3S2eipQDd3Lysqr9w4anZOjuHEu9UjZKle2Fji7UgWHip2r0zK2bNNka4m2/cgYDnBIQ5czyTq+9zbioe4g5WsPWKauLJnyENBFzwV2mnyWc4rMgY4C7/WOv9FfpmgnRpHiiKakdZtpZczjlaw1EoZGL7XW65pFtFmVuD1lW576WkfKNrggC/vK0y3pHlIwIqknXdalHUluoSf8Ax/F4NDhk5H4LO+RVqDC8Wbp+jakDkYyB8VlJT88TMQmd1DTffdY133Pa4ldA/AsYqoQwxxU4vvJJ+Tbo4uh8g/b4g0cwyMm/nZXd1leTGMKKR7fWKbUTH0Z1vuroo+itKz9pUVLv3SG/VO/QWGNbldT9Z+/I4/KycliPmxjjIZc291ZExIXVHC6AerRxD3IThtEP/mi8kHP5Eck5wOtwgGR59ZVg4uG6gXbnIKzPZskLHF1nAi3ksSRhjmLWm9jutF0jmMzE7KnC0vOZ25QJ2v4fJ2hc6rpKMmwJXPUsQzNsNV0VFC52UXSnttJ01qdwIBIsr8cws0LOjiDLBzr9yv04a3S39FrBYMh79hZBJTu6t1yduStNIBsi0c08lWme2YKcEXJOioVUJe/Q9j5rWmeD9m33uCRPGdLAbIuI2x3RiJhcWkk/NKih/wAwuBLtSFYrj+sNj4NGY96a6BwbdoWa4QwtBsrlO5oSGwuOptdSczNLFOTtW1yapMcEkkbGuLWm2vFUqfpJiLAyICma1oA/Znf+ZUsWq5IYRl2B1WIysc5+bnqjLLtPhL7d/S41Vzev1B8GkfmVZkxGeNud1OxzBxa61vguSw2rdZq16yuDaTKCNVeOW2WXFj9NEY0x28T/AOYKRibHbxP8x9Vx7a111ZbVvs3VF0V4Y6WTEYWgFzZG35i6UcUgOzZPIfVc9W4gckY43Kqmsc5mmif9ROGV07a9knqtcPJOLXHUArmKGre2TXVdXST5YQLA68UrYeXDNPlfpVtAjZUgtf4L6nN0dwae+fDKe545bLOqeg2CzAmJs1O4/ckuPJLxc85Hzatn+xFuKXSy9oC667FP8OanqicOr2yW1yTNs53dcaBcfUUdVhdQYa2B8LwbWeLX8DxUWNcc5WzSv7YXQUMthoQuVo5wLXIWuycFoA05lT6dWN6b8VR1kg17gFoRzXAF9VzVNO3rAb7K/T1O2qqU622PN90c8rhGAw9o7rPiqh7WwG6KKQEF5vcq9s9LhLWALx7QJ7tFXMmdw1Sq2dsUT3A6AbpyiTtmseJK58hNxe3uC1gRYAarnaSTY8StFk2g0UY1dxX2svJonyx5Y7CMknTTVIpHA2J5LG6QzzVEjWRzviYx1+w4jN7wq8isv09jmE47IC2mwpzgNzmFz4Bc6+jxClcDV0dREeRiJ+S6fDq2usGjEJ9PxXv5rXfiuIwQh2eOZo3a8a+aXj5dsrc8XEUtUIndolvjorFZizMjWBzT7117Mchez9Yw+Ik7hoafmFEmJ4fEwubhrQB+BiJjSueV+nEx1LTxCuU9QHLphjVE4XFC0fwtRMxWnJs2kZc9wCJL9n5ZX1i5eua+RzQxjz4NKCOCctsIZP5Su6hl61maNpaOQXnE9/uKdiLy54dWOPpKGp61p6mTy0XQQxyNjAkf2u5WnOJ3PwSidUkZc2WTeBRJd0QK0cw7pFdQ0uI07oK6Bk0bvZcNvfwTLogU9bH+OHxL/D9ou7CKnLxEU+o9zvqufq8HxnDiTU0khY3XPH22/DX4L6zfVFe4tYKbhGmPLlHx2CtuCAdlbp602FnXsvo1dgmG1/8A7NHE4/eDbHzCyZeg+GEkwT1EPIBwd81Hg2n8n9c4Kw9WNd1birhk9YLQk6EH/JxIj9+O9vJIk6G1rTeOshf4tLfzR42LnPjQwVIc7UqrjVVaAsDvWKvx9F8UZtLT/wA5+iRWdEsUqJGu6ymsBtnP0RZdH82DIpXm60YJQd7KzB0SxJre1LTj+In8k2HonXhxL6mBovwuUpiq8+ADUCKLNmA0NlgVFSJn7rp6vofNPE1oxLK4bjq9PmsmToTicesM9PKO9xafzRcaU58VOll6s9kFWqyskEGW/BHF0cxiLR0EZ72vuqdZg2NslGehkLObCCPmiSyHeXCksqX80dXVv9GI43QChrmnWjqB/tOT5MLxGohLYqOUnkQB80TY+TFQjrHZbJkFZIJm6p8PRzFPapHN/ecPyKu03Rira7NI6JvvuUdqnLhHQYPU2ZldrorTiqlFQijZ6+d/Paye4qnNzckzvQHFJJ1RvKSTqkxdBdTdLuputozMuUQKVmUgoBt0V0oFFdFA7qboLr10jHdeugzd69mQB3XroLr2ZAHdeugzKMyQHmUEoLqCUAeZAT3oSUJckYnPPMpbncyoLkDigPOcUpxUuKU5yDiHFJcUT3JLnJUBeUknVE9ySXaqTdCCpBSQ/RSHLZmcCiBSMyIOQDgUV0kOU5kA7MoLkrMvZkUG3U5knMpzJGbmUXQZlGZAMuoul5lGZIGlyguS8yguQBlyAuQl6AvSMRcgc5C56U56AMuSnOQl6U56DgnOSXuUPekvelQ896SX6qHvSi7VI3//2Q==")
    image = imutils.resize(image,width=250)
    target_image = imutils.resize(target_image,width=250)


    # plt.subplot(3,1,1)
    # plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # plt.title("original Image")
    # plt.show()

    #Apply Skin Mask
    skin = createSkin(image)
    # skin = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
    # skin_color = skin.copy()
    #
    #
    target_skin = createSkin(target_image)
    # target_skin = cv2.cvtColor(target_skin, cv2.COLOR_BGR2RGB)
    # target_skin_color = target_skin.copy()
    # red_diff = np.subtract(target_skin_color[:,:,0], skin_color[:,:,0])
    # print(target_skin_color[:,:,0][125][0])
    # plt.subplot(3, 1, 2)
    # plt.imshow(target_skin_color[:,:,0])
    # plt.imshow(target_skin_color[:,:,1])
    # plt.imshow(target_skin_color[:,:,2])
    # plt.title("Thresholded Image")
    # plt.show()

    transfered_color = skin_color_transfer(skin,target_skin,clip=True,preserve_paper=True)

    show_image("transfered",transfered_color)