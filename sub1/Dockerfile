FROM jupyter/scipy-notebook
FROM python:3.9-slim-buster

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN pip install joblib
RUN pip install Pandas
RUN pip install Numpy
RUN pip install onnxruntime
RUN pip install audio2numpy
RUN pip install argparse
RUN pip install librosa pydub

COPY * /
RUN pip install scipy lightgbm onnxruntime

ENTRYPOINT ["sh", "predict.sh"]
