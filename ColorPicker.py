import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)



def compare_values_in_hsv(img_original, img_attack, img_fingerprint):
    """
    Compares the brightness levels of an regular image of an roi and an attack image of an roi
    :param img_original: the roi of the original image ndarray(600x600)
    :param img_attack: the roi of the attack image ndarray(600x600)
    :param img_fingerprint: the corresponding binariesed fingerprint of the original ndarray(540x540)
    :return:
    """
    img = img_attack
    original = img_original
    original = cv2.resize(original, (600, 600))
    # Get the Fingerprint (Mask for all bright spots on Attack)
    fingerprint = img_fingerprint
    fingerprint = cv2.cvtColor(fingerprint, cv2.COLOR_BGR2GRAY)
    # Invert the Fingerprint (Mask for all dark spots on Attack)
    fingerprint_inverted = (255 - fingerprint)
    fingerprint_AND_Attack = cv2.bitwise_and(img[30:570, 30:570], img[30:570, 30:570], mask=fingerprint)
    fingerprint_inverted_AND_Attack = cv2.bitwise_and(img[30:570, 30:570], img[30:570, 30:570],
                                                      mask=fingerprint_inverted)
    # Convert back to HSV to compare brightness values
    fingerprint_AND_Attack_HSV = cv2.cvtColor(fingerprint_AND_Attack, cv2.COLOR_BGR2HSV)
    fingerprint_inverted_AND_Attack_HSV = cv2.cvtColor(fingerprint_inverted_AND_Attack, cv2.COLOR_BGR2HSV)
    # [:,:,2] is the Value in HSV (Brightness level)
    attack_bright = np.round(fingerprint_AND_Attack_HSV[:,:,2].sum()/np.count_nonzero(fingerprint),4)
    print("Bright Spots Attack:", attack_bright)
    attack_dark = np.round(fingerprint_inverted_AND_Attack_HSV[:,:,2].sum()/np.count_nonzero(fingerprint_inverted),4)
    print("Dark Spots Attack:", attack_dark)
    # Same thing for The Original Image
    fingerprint_AND_Original = cv2.bitwise_and(original[30:570, 30:570], original[30:570, 30:570], mask=fingerprint)
    fingerprint_inverted_And_original = cv2.bitwise_and(original[30:570, 30:570], original[30:570, 30:570], mask=fingerprint_inverted)
    fingerprint_AND_Original_HSV = cv2.cvtColor(fingerprint_AND_Original, cv2.COLOR_BGR2HSV)
    fingerprint_inverted_And_original_HSV = cv2.cvtColor(fingerprint_inverted_And_original, cv2.COLOR_BGR2HSV)
    original_bright = np.round((fingerprint_AND_Original_HSV[:,:,2].sum()/np.count_nonzero(fingerprint)),4)
    print("Bright Spots Original:", original_bright)
    original_dark = np.round((fingerprint_inverted_And_original_HSV[:,:,2].sum()/np.count_nonzero(fingerprint_inverted)),4)
    print("Dark Spots Original:", original_dark)
    #fig, ax = plt.subplots()
    #X_Axis = np.arange((4))
    #ax.bar(X_Axis,(attack_bright,attack_dark,original_bright,original_dark),0.35,color='r')
    #ax.set_xticks(X_Axis + 0.35/2)
    #ax.set_xticklabels(("attack_bright","attack_dark","original_bright","original_dark"))
    #ax.set_ylabel("Mean Value in HSV")
    #plt.draw()
    #plt.show()
    return ((original_bright - original_dark)/original_dark)*100, ((attack_bright - attack_dark)/attack_dark)*100

# img = cv2.imread("./Real_Attack_imgs/1.png")
# cv2.imshow("Img",img)
# cv2.waitKey(3000)
original = cv2.imread("./Original_Image/roi_mapped_post.png")
fingerprint = cv2.imread("./Fingerprints/3added_5_mask.png")

all_diff = []
is_Attack = []

title = "Magenta-Global"
path = "./Real_Attack_Imgs_Red"
for count, test_file_name in enumerate(os.listdir(path)):
    print(test_file_name)
    if (test_file_name.split('.')[-1] == 'png'):
        img = cv2.imread(os.path.join(path,test_file_name))
        cv2.imshow("IMG",img)
        cv2.waitKey(1000)
        #print("Image Nr:",count)
        realtive_difference_original,realtive_difference_attack = compare_values_in_hsv(img_original=original,img_attack=img,img_fingerprint=fingerprint)
        print(realtive_difference_original,realtive_difference_attack)
        all_diff.append(realtive_difference_attack)
        is_Attack.append("Yes")

path = "./Original_Image"
for count, test_file_name in enumerate(os.listdir(path)):
    print(test_file_name)
    if (test_file_name.split('.')[-1] == 'png'):
        img = cv2.imread(os.path.join(path,test_file_name))
        cv2.imshow("IMG",img)
        cv2.waitKey(100)
        #print("Image Nr:",count)
        realtive_difference_original,realtive_difference_attack = compare_values_in_hsv(img_original=original,img_attack=img,img_fingerprint=fingerprint)
        print(realtive_difference_original,realtive_difference_attack)
        all_diff.append(realtive_difference_original)
        is_Attack.append("No")

df = pd.DataFrame(list(zip(all_diff, is_Attack)),
               columns =['relative_diff', 'is_Attack'])

sns.boxplot(x='is_Attack',y="relative_diff", hue="is_Attack",
                 data=df, palette="Set3").set_title(title)
plt.title = title
plt.savefig(os.path.join("./Plots",title + ".png"))
plt.draw()
plt.show()

print(df.head(20))
while True:
    """
    #success, img = cap.read()
    img = cv2.imread("BHO#5#30.png")
    original = cv2.imread("Original.PNG")
    fingerprint = cv2.imread("./Fingerprints/BHO#5#CCClassic_5_mask.png")
    #compare_values_in_hsv(img_original=original,img_attack=img,img_fingerprint=fingerprint)
    
    original = cv2.resize(original,(600,600))
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    originalHSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    # Get the Fingerprint ( Mask for all bright spots on Attack)
    fingerprint = cv2.cvtColor(fingerprint,cv2.COLOR_BGR2GRAY)
    # Invert the Fingerprint (Mask for all dark spots on Attack)
    fingerprint_inverted = (255-fingerprint)
    #cv2.imshow("Fingerprint",fingerprint)
    #print(fingerprint.shape)
    # Pad both Fingerprints to match the shape
    fingerprintPAD = np.pad(fingerprint,(30,30),mode='maximum')
    fingerprint_invertedPAD = np.pad(fingerprint_inverted,(30,30),mode='maximum')
    cv2.imshow("Fingerprint_INVERTED",fingerprint_invertedPAD)
    cv2.imshow("FingerprintPAD",fingerprintPAD)

    #print(imgHsv.shape)
    #print("Original SDT:", originalHSV[:,:,2].std())

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    #print(h_min)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    #print("Mask SDT:",mask[:,:,2].std())
    #print("Attack STD:",result[:,:,2].std())
    print("Original HSV",originalHSV[:,:,2].mean())
    # Bitwise And the Attack Image with the Mask and inverse Mask
    fingerprint_AND_Attack = cv2.bitwise_and(img[30:570,30:570], img[30:570,30:570], mask=fingerprint)
    fingerprint_inverted_AND_Attack = cv2.bitwise_and(img[30:570,30:570], img[30:570,30:570], mask=fingerprint_inverted)
    #cv2.imshow("Fingerprint AND Attack" , fingerprint_AND_Attack)
    #cv2.imshow("Fingrprint_inverted AND Attack", fingerprint_inverted_AND_Attack)

    # Convert back to HSV to compare brightness values
    print(fingerprint.shape)
    fingerprint_AND_Attack_HSV = cv2.cvtColor(fingerprint_AND_Attack,cv2.COLOR_BGR2HSV)
    fingerprint_inverted_AND_Attack_HSV = cv2.cvtColor(fingerprint_inverted_AND_Attack,cv2.COLOR_BGR2HSV)
    # Calculate the correct HSV Values
    attack_bright = np.round(fingerprint_AND_Attack_HSV[:,:,2].mean()/np.count_nonzero(fingerprint)*291600)
    print("Bright Spots Attack:", attack_bright)
    attack_dark = np.round(fingerprint_inverted_AND_Attack_HSV[:,:,2].mean()/np.count_nonzero(fingerprint_inverted)*291600)
    print("Dark Spots Attack:", attack_dark)

    # See the same Results on an non Attack Image:
    a = cv2.bitwise_and(original[30:570,30:570], original[30:570,30:570], mask=fingerprint)
    b = cv2.bitwise_and(original[30:570,30:570], original[30:570,30:570], mask=fingerprint_inverted)
    a_HSV = cv2.cvtColor(a,cv2.COLOR_BGR2HSV)
    b_HSV = cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
    original_bright = np.round((a_HSV[:,:,2].mean()/np.count_nonzero(fingerprint)*291600))
    print("Bright Spots Original:", original_bright)
    original_dark = np.round((b_HSV[:,:,2].mean()/np.count_nonzero(fingerprint_inverted)*291600))
    print("Dark Spots Original:", original_dark)
    print("#####################")
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack1 = np.hstack([original, img])
    hStack3 = np.hstack([fingerprint_AND_Attack,fingerprint_inverted_AND_Attack])
    hstack4 = np.hstack([a,b])
    hStack2 = np.hstack([mask, result])
    #vStack = np.vstack([hStack1,hStack3])
    cv2.imshow('Horizontal Stacking', hStack1)
    cv2.imshow('Attack with Mask', hStack3)
    cv2.imshow('Original with Mask', hstack4)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """
cap.release()
cv2.destroyAllWindows()