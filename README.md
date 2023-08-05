# Customer Segmentation 

This project focuses on creating a machine learning model for customer segmentation using the K-Means Clustering algorithm. The objective is to group customers based on their and other relevant features. Additionally, a web application using Streamlit is developed to visualize the segmentation results.

# Table of Contents

- Project Overview
- Installation
- Usage
  
## Project Overview

In this project, we harness the capabilities of K-Means Clustering, a widely used unsupervised learning algorithm, to categorize customers into distinct segments based on their shared characteristics. This approach can provide valuable insights for targeted marketing, personalized recommendations, and business strategy optimization.

The project repository includes the following components:

App/: This directory holds the source code files for the Flask web application.
customer_data.csv/: the dataset used for training and testing the segmentation model.
clustering.pkl: The serialized K-Means Clustering model.
streamlit.py: The Streamlit application for deploying the trained model and visualizing segment results.
requirements.txt: A list of Python dependencies required to run the project.
Customer_Segmentation.ipynb: The Jupyter Notebook used for building and analyzing the segmentation model.

The Machine Learning Model

The customer segmentation model follows these steps:

        -Load and preprocess the customer dataset.
        -Apply K-Means Clustering to group customers into segments.
        -Serialize and save the trained clustering model.
## Installation

To set up the project environment, follow these steps:

Clone the repository:
[git clone https://github.com/your-username/customer-segmentation.git]

Navigate to the project directory:
[cd Customer-Segmentation]

Install the dependencies:
[pip install -r requirements.txt]

# Usage

Run the streamlit application:

[streamlit run streamlit.py]

Access the web application through your browser at http://localhost:5000. The user interface will allow you to upload a CSV file containing customer data.

Fill in the necessary details about the customers and submit the form. The application will process the data and display the customer segment
