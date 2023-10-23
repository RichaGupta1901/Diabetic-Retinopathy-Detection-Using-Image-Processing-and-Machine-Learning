import streamlit as st
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
from skimage.morphology import skeletonize

pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)



def optic_disc(image):
    ratio  = min([1152/image.shape[0], 1500/image.shape[1]])
    img = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
    image_r = img[:, :, 2]

    threshold_value = 245
    _, thresh = cv2.threshold(image_r, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_opening = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=2)

    kernel_closing = np.ones((10, 10), np.uint8)
    closed_optic_disc = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel_closing, iterations=2)

    contours, _ = cv2.findContours(closed_optic_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) > 0:
        largest_contour = contours[0]
        optic_disc_mask = np.zeros_like(image_r)
        cv2.drawContours(optic_disc_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        optic_disc_image = cv2.bitwise_and(image_r, image_r, mask=optic_disc_mask)
        smooth_optic_disc = cv2.GaussianBlur(optic_disc_image, (0, 0), sigmaX=5, sigmaY=5)
        kernel = np.ones((10, 10), np.uint8)
        dilated_optic_disc = cv2.dilate(smooth_optic_disc, kernel, iterations=1)
        _, binary_dilated_optic_disc = cv2.threshold(dilated_optic_disc, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_dilated_optic_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_optic_disc = dilated_optic_disc.copy()
        cv2.drawContours(filled_optic_disc, contours, -1, 255, thickness=cv2.FILLED)

        return filled_optic_disc
    else:
        print("No contours found. Returning default image.")
        return np.zeros_like(image_r, dtype=np.uint8)

def adjust_gamma(image, gamma=1.0):
   table = np.array([((i / 255.0) ** gamma) * 255
   for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

def extract_ma(image):
    r,g,b=cv2.split(image)
    comp=255-g
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    histe=clahe.apply(comp)
    adjustImage = adjust_gamma(histe,gamma=3)
    comp = 255-adjustImage
    J =  adjust_gamma(comp,gamma=4)
    J = 255-J
    J = adjust_gamma(J,gamma=4)
    
    K=np.ones((11,11),np.float32)
    L = cv2.filter2D(J,-1,K)
    
    ret3,thresh2 = cv2.threshold(L,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel2=np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    kernel3=np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    return opening

def blood_vessel(image):
    ratio = min([1152 / image.shape[0], 1500 / image.shape[1]])
    resized_image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(resized_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green_channel = clahe.apply(g)
    
    img_medf = cv2.medianBlur(enhanced_green_channel, 131)
    img_sub = cv2.subtract(img_medf, enhanced_green_channel)
    img_subf = cv2.blur(img_sub, (7, 7))
    ret, img_darkf = cv2.threshold(img_subf, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_darkl = cv2.morphologyEx(img_darkf, cv2.MORPH_OPEN, kernel)

    img_medf1 = cv2.medianBlur(enhanced_green_channel, 131)
    img_sub1 = cv2.subtract(img_medf1, enhanced_green_channel)
    img_subf1 = cv2.blur(img_sub1, (7, 7))
    ret, img_darkf1 = cv2.threshold(img_subf1, 12, 255, cv2.THRESH_BINARY)
    img_darkl1 = cv2.morphologyEx(img_darkf1, cv2.MORPH_OPEN, kernel)

    img_both = cv2.bitwise_or(img_darkl, img_darkl1)

    result = cv2.resize(img_both, (enhanced_green_channel.shape[1], enhanced_green_channel.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    return result

def exudate_extraction(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   
    total_x_position = 0
    total_white_pixels = 0
    total_y_position = 0
    sum_values=0
    v,k=image.shape
    for y in range(len(image)):
        for x in range(len(image)):
            sum_values+=image[y][x]
    avg=sum_values/(v*k)
    
    threshh = (2.3547*avg) + 10.292
    _, binary_image = cv2.threshold(image, threshh, 255, cv2.THRESH_BINARY) #148

    for y in range(len(binary_image)):
        for x in range(len(binary_image)):
            if binary_image[y][x] == 255:
                total_x_position += x
                total_y_position += y
                total_white_pixels += 1
                
     
    if total_white_pixels == 0:
        average_x_position = 0
        average_y_position=0
    else:
        average_x_position = total_x_position / total_white_pixels
        average_y_position = total_y_position / total_white_pixels
    if average_y_position<1000:
        for j in range(binary_image.shape[0]):
            for i1 in range(binary_image.shape[1]):
                if j<(average_y_position+300):
                    binary_image[j,i1]=0
    if average_x_position>2200 or average_x_position==0:
        for j in range(binary_image.shape[0]):
            for i1 in range(binary_image.shape[1]):
                if i1>(average_x_position-240):
                    binary_image[j,i1]=0
    else:
        for j in range(binary_image.shape[0]):
            for i1 in range(binary_image.shape[1]):
                if i1<(average_x_position+240): 
                    binary_image[j,i1]=0

    return binary_image
    
def Ratio_MA(image):
    num_white_pixels = cv2.countNonZero(image)
    total_pixels = image.size
    ratio = num_white_pixels / total_pixels

    if ratio:
        return ratio
    else: 
        return 0
    
def blood_vessel_length(image):
    vessel_skeleton = skeletonize(image)
    length = np.sum(vessel_skeleton)
    if length:
        return length
    else:
        return 0
    
    
def blood_vessel_tortuosity(image):
    vessel_skeleton = skeletonize(image)
    vessel_length = np.sum(vessel_skeleton)
        
    contours, _ = cv2.findContours(vessel_skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    tortuosity = perimeter / vessel_length

    if tortuosity:
        return tortuosity
    else:
        return 0
    
def mean_intensity_image(image):
    mean_intensity = np.mean(image)
    return mean_intensity

def original_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def about():
    st.header("About")

    st.write("Our motivation for the project is to reduce the time consumed in extracting clinically important features from the Medical images generated using an Ophthalmoscope for fundus imaging. The early detection of diabetic retinopathy using the extracted features by Image Segmentation can lead to early intervention that can prevent the loss of vision due to DR at its earliest stage called Micro aneurysms (MA’s).")

    st.write("Our project aims at developing an Image Segmentation model which refers to automating the process of extracting features that detect Diabetic retinopathy in retinal images. Image Segmentation helps us understand the image content for searching and mining in medical image archives.")

    st.markdown("### The Difference between the vision and characteristics of a normal eye and a Diabetic Retinopathic eye:")

    img1 = Image.open("DR_comparison.png")
    st.image(img1, caption="Retina Comaprison (Normal eye vs. DR eye)", use_column_width=True)

    img2 = Image.open("Vision_comparison.png")
    st.image(img2, caption="Vision Comaprison",use_column_width=True)

def classifying(ma_ratio,bv_length,bv_tortuosity,mean_intensity):

    features = [ma_ratio, bv_length, bv_tortuosity, mean_intensity]
    features = np.array(features).reshape(1, -1)  # Reshape the features into a 2D array

    pred = model.predict(features)
    print(pred)
    return pred
    

def main():
    st.title("Diabetic Retinopathy Prediction")

    selected_segment = st.sidebar.radio("Select a further action", 
                                        ("Prediction and Segmentations", "About"))
    
    if selected_segment=="Prediction and Segmentations":

        st.header("What is Diabetic Retinopathy?")
        st.write("Diabetic Retinopathy (DR) is a leading cause of preventable impairment of vision in people of the working age . This condition is observed among people with diabetes. It affects the blood vessels and the light-sensitive tissues of the retina of the eye.")

        upload_image = st.file_uploader('Upload the image of retina here:', type=["jpg", "jpeg", "png"])

        if upload_image:
            st.header("Image provided:")
            file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            og_image = original_image(image)
            st.image(og_image, caption='Uploaded Image', use_column_width=True)
            
            st.header("Optic Disc:")
            optic_disc_image = optic_disc(image)
            od_processed_image = Image.fromarray(optic_disc_image)
            st.image(od_processed_image, caption='Optic Disc Image', use_column_width=True)

            st.header("Microaneurysms:")
            ma_image = extract_ma(image)
            ma_processed_image = Image.fromarray(ma_image)
            st.image(ma_processed_image, caption="Microaneurysms Image", use_column_width=True)

            ma_ratio = Ratio_MA(ma_image)

            st.header("Blood Vessels:")
            bv_image = blood_vessel(image)
            bv_processed_image = Image.fromarray(bv_image)
            st.image(bv_processed_image, caption="Blood Vessels Image", use_column_width=True)

            bv_length = blood_vessel_length(bv_image)
            bv_tortuosity = blood_vessel_tortuosity(bv_image)

            mean_intensity = mean_intensity_image(image)

            st.header("Exudates:")
            exudates_image = exudate_extraction(image)
            exudates_processed_image = Image.fromarray(exudates_image)
            st.image(exudates_processed_image, caption="Exudates Image", use_column_width=True)

            if st.button("Predict"):
                result = classifying(ma_ratio,bv_length,bv_tortuosity,mean_intensity)
            
                if result == 1:
                    st.success('This Image shows Diabetic Retinopathy is Present.')
                else:
                    st.success('This Image does not show signs of Diabetic Retinopathy.')
                

    if selected_segment=="About":
        about()


if __name__ == '__main__':
    main()
        
