FROM python:3.9.19-alpine3.20

COPY . /Graphical-Sentiment-Analysis

WORKDIR /Graphical-Sentiment-Analysis

# pip commands
RUN pip install --upgradable pip &&\
    pip install -r ./requirement.txt &&\
    python -m spacy download en_core_web_sm

# fastext installation
RUN sudo apt install git &&\
    git clone https://github.com/facebookresearch/fastText.git &&\
    pip install ./fastext/ &&\ 
    rm -rf ./fastext

# fasttext model training initialization
RUN python3 ./ft-training.py

# training
CMD python3 ./training.py
