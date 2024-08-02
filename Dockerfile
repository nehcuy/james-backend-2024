# Use the official Python image from the Docker Hub
FROM python:3.12-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Flask will run on
EXPOSE 5000

# Define environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask application
CMD ["python", "-u", "main.py"]
