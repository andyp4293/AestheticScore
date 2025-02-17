# AestheticScore

AestheticScore is a machine learning-powered web application that predicts attractiveness scores based on facial images. The model is trained on a dataset of 5,500+ images of Caucasian and Asian men and women, each labeled with an attractiveness rating from 1 to 5.

## Features
- **FastAPI Backend**: Processes image uploads and runs inference using a trained deep learning model.
- **React Frontend (Netlify)**: Provides an intuitive UI for users to upload images and receive attractiveness scores.
- **Machine Learning Model**: Trained using PyTorch with a CNN-based architecture for image classification.
- **AWS Deployment**: Hosted on an AWS EC2 instance with an Nginx reverse proxy for HTTPS.
- **CORS Configured**: Ensures cross-origin requests are allowed between frontend and backend.

## Tech Stack
- **Backend**: FastAPI, Python, PyTorch
- **Frontend**: Angular
- **Cloud & Deployment**: AWS EC2, Netlify

## Setup Instructions

### Backend (FastAPI)
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AestheticScore.git
   cd AestheticScore/aestheticscore-backend
