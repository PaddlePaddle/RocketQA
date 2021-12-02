FROM paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7

# setup the workspace
COPY . /rocketqa
WORKDIR /rocketqa

ENV JINA_USE_CUDA=True

# install dependency
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]