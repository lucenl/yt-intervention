# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set current directory as ENV
ENV PATH=/app:$PATH

# Needed for tzdata
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV DISPLAY=:99

# copy items
WORKDIR /app
# COPY data/training/ ./data/training/
# COPY arguments/harmful_50_1,38eb77f8.json ./args/harmful_50_1,38eb77f8.json

# Install pre-dependencies
RUN apt update
RUN apt install -y wget g++ unzip xvfb firefox-esr git

# install google chrome
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb

# install python dependencies
RUN pip install --upgrade git+https://github.com/Haroon96/YouTube-Driver.git
COPY requirements.txt ./
RUN pip install -r requirements.txt

# copy code files
COPY sockpuppet.py ./