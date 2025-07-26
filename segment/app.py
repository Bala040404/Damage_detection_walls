from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Update the model path to a local directory
model_path = "./segmentation_model/weights/best.pt"  # Replace with the local path to your best.pt
model = YOLO(model_path)


# Update the image path to a local directory
image_path = r"E:\Final year project\vegitation\cracks_al_aqsa.jpg"  # Replace with the full path to your image

# Perform prediction without saving results
results = model.predict(source=image_path, save=False)

# Get the image with the segmentation overlay
# Assuming the results object has an 'imgs' attribute with the processed images
segmented_image = results[0].plot()  # Access the first result and plot the segmentation

# Display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(
    cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
)  # Convert BGR to RGB for correct color display
plt.axis("off")
plt.title("Segmented Image")
plt.show()
