FROM paddlepaddle/paddle:2.2.0

# setup the workspace
COPY . /rocketqa
WORKDIR /rocketqa

# install dependency
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]