# Use official python runtime as a parent image
FROM python:3.7.3-stretch
MAINTAINER Pierre-Luc Delisle, github@pierre-luc-delisle.com

# Set the working directory to /app.
WORKDIR /app

# Copy the current directory contents into the container at /app.
ADD . /app

# 1) Upgrade pip.
# 2) Install any needed packages specified in requirements.txt.
# 3) Install this package into the container.
RUN pip install --upgrade pip && \
    pip install --trusted-host pypi.python.org -r requirements.txt && \
    python setup.py install