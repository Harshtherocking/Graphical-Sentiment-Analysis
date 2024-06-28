FROM ubuntu:latest

COPY . /Graphical-Sentiment-Analysis

WORKDIR /Graphical-Sentiment-Analysis

# git and python installation
RUN apt update --yes &&\
    apt install --yes build-essential &&\
    apt install --yes git &&\
    apt install --yes python3
  
# pip installation
RUN apt install --yes python3-venv python3-pip

# virtual environment 

# package installation

RUN pip install --break-system-packages -r ./requirement.txt &&\
    python3 -m spacy download en_core_web_sm

# fastext installation
RUN git clone https://github.com/facebookresearch/fastText.git &&\
    pip install --break-system-packages ./fastext/ &&\ 
    rm -rf ./fastext

# fasttext model training initialization
RUN python3 ./ft-training.py

# training
CMD python3 ./training.py
