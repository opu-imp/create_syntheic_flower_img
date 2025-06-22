FROM python:3.10-slim

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install linux packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /work