# image-video-analysis-project-1
Image and Video Analysis project 1 repository.
Team: Josep and Mira

The aim of the project is to estimate the number of people present on a beach using traditional Computer Vision techniques.

"imgs" folder contains the images and labels of manually annonated images. 
"metrics.py" contains the function to calculate the MSE and accuracy metrics (Precision, Recall, F1-Score and Accuracy).

To run the code, download the ZIP file and run the "solution_josep.ipynb" file.
In the code, you can choose one out of the 3 pre-processing options: Gaussian Blur with Canny edge detection, Laplacian filter, or Histogram equalization.

# Results
## Applying Gaussian Blur with Canny edge detection provide the best performance out of all options:
![image](https://github.com/user-attachments/assets/2de80eb6-da1e-4b3f-9d27-2c2b2a6c6e24)
![image](https://github.com/user-attachments/assets/7ffd3452-cc85-4f40-b47f-5ccbcbc3ca33)



## Laplacian (High-pass filtering) shows lower accuracy:
![image](https://github.com/user-attachments/assets/1700a3ad-f831-42a2-ac87-de61e3d4e8e7)
![image](https://github.com/user-attachments/assets/887b8d61-7613-44fd-bc4f-b5b69e20deda)

## Histogram qualization shows the worst accuracy:
![image](https://github.com/user-attachments/assets/db1b0371-b493-4bc7-80e8-f7978af1c9c1)
![image](https://github.com/user-attachments/assets/40afc9a4-0a75-436b-9a70-b048597344da)




