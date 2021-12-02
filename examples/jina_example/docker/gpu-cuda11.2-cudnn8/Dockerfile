FROM paddlepaddle/paddle:2.2.0-gpu-cuda11.2-cudnn8

# setup the workspace
COPY . /rocketqa
WORKDIR /rocketqa

ENV JINA_USE_CUDA=True

# install dependency
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]