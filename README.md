# **Customer Segmentation Using KMeans Clustering**

## **Project Overview**
This project implements a machine learning model to perform customer segmentation based on their demographic and spending behavior. Using the KMeans clustering algorithm, customers are grouped into distinct clusters, such as "Low Income, Low Spending," "High Income, High Spending," and "Moderate Income, Moderate Spending." The project also includes an interactive web application built with **Streamlit**, allowing users to input customer details and predict their cluster in real time.

---

## **Features**
- Train a KMeans clustering model using predefined customer data.
- Interactive web interface to predict the cluster of a new customer based on:
  - Gender
  - Age
  - Annual Income
  - Spending Score
- Display meaningful cluster descriptions instead of numeric IDs.

---

## **Technologies Used**
- **Python**: For data processing and model training.
- **scikit-learn**: For implementing the KMeans clustering algorithm.
- **pandas**: For data manipulation and analysis.
- **Streamlit**: For building the interactive user interface.

---

## **How to Run the Project**

### **1. Prerequisites**
Ensure you have Python installed on your system. Install the required libraries using:
```bash
pip install streamlit scikit-learn pandas numpy
