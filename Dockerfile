# Use TensorFlow's official Docker image as a base image (with Python 3.9)
FROM tensorflow/tensorflow:2.6.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install virtualenv (if using Python's built-in venv, skip this step)
RUN pip install virtualenv

# Create a virtual environment in the /app/venv directory
RUN python3 -m venv /app/venv

# Activate the virtual environment and install other dependencies from requirements.txt
# Use the TensorFlow pre-installed in the base image
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Set the environment variables to ensure the virtual environment is used
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

# Expose port 5000 to access the web app
EXPOSE 5000

# Run the Flask app using the virtual environment's Python
CMD ["/app/venv/bin/python", "predictions.py"]


