# Customer Segmentation 

This project focuses on creating a machine learning model for customer segmentation using the K-Means Clustering algorithm. The objective is to group customers based on the total bytes consumed by their devices, the downlink throughput and other relevant features. Additionally, a web application using Streamlit is developed to visualize the segmentation results.

# Table of Contents

- Project Overview
- Installation
- Usage
  
## Project Overview

In this project, we harness the capabilities of K-Means Clustering, a widely used unsupervised learning algorithm, to categorize customers into distinct segments based on their shared characteristics. This approach can provide valuable insights for targeted marketing, personalized recommendations, and business strategy optimization.

The project repository includes the following components:

App/: This directory holds the source code files for the Flask web application.
clustering.pkl: The serialized K-Means Clustering model.
streamlit.py: The Streamlit application for deploying the trained model and visualizing segment results.
requirements.txt: A list of Python dependencies required to run the project.
Customer_Segmentation.ipynb: The Jupyter Notebook used for building and analyzing the segmentation model.


The customer segmentation model follows these steps:

        -Load and preprocess the customer dataset.
        -Apply K-Means Clustering to group customers into segments.
        -Serialize and save the trained clustering model.
        
## Installation

To set up the project environment, follow these steps:

Clone the repository:
git clone https://github.com/Nouhailangr/customer-segmentation.git

Navigate to the project directory:
cd Customer-Segmentation

Install the dependencies:
pip install -r requirements.txt

# Usage

Run the streamlit application:

streamlit run streamlit.py

<img width="750" alt="Screenshot 2023-08-05 at 19 44 17" src="https://github.com/Nouhailangr/Customer-segmentation/assets/127351602/7d74976d-8e47-419a-b2d1-600e34f72a27">

Fill in the necessary details about the customers and submit the form. The application will process the data and display the customer segment by clicking "Customer segmentation" Button.

<img width="750" alt="Screenshot 2023-08-05 at 19 45 10" src="https://github.com/Nouhailangr/Customer-segmentation/assets/127351602/1e5996e1-d62f-4c3d-a654-977890960865">
