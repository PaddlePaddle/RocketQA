# Jina❤️RocketQA

This is a simple Demo of how to use RocketQA together with [Jina](http://get.jina.ai/). Jina is a Cloud-native neural search framework for _**any**_ kind of data

## Run Quick Demos

> **Prerequisites**: Make sure `docker` is installed and running on your machine


```shell  
# delete this line once the docker image is published
docker build -t rocketqa-jina -f docker/cpu/Dockerfile .
docker run --rm -it -v "$(pwd)/workspace:/rocketqa/workspace" -v "$(pwd)/model:/root/.rocketqa" rocketqa-jina:latest index
```

The above codes will download the models to `model` folder and index the data at `toy_data/test.tsv` from the Docker image. The index will be stored at `workspace` folder. You can start asking questions via your browser by running the following command and open `workspace/static/index.html` in your browser.

```shell
docker run --rm -it -v "$(pwd)/workspace:/rocketqa/workspace" -v "$(pwd)/model:/root/.rocketqa" -p 8886:8886 rocketqa-jina:latest query
```

If you prefer to use CLI, please run

```shell
docker run --rm -it -v "$(pwd)/workspace:/rocketqa/workspace" -v "$(pwd)/model:/root/.rocketqa" rocketqa-jina:latest query_cli
```


## Usages

### Install Dependencies
```shell
cd examples/jina_example
pip install -r requirements.txt
```

### Index

Run the following line to index the data stored at `toy_data/test.tsv`

```shell
python app.py index toy_data/test.tsv
```

The data is formatted as below
```text
title_1\ttext_1\n
title_2\ttext_2\n
...
```

### Query

Run the following line to start the query service. You will have a demo page opened in your browser. By default, port `8889` is used. 

```shell
python app.py query
```

If you prefer to use the shell, please try

```shell
python app.py query_cli
```


### Tips

If you have GPUs, please set `export JINA_USE_CUDA=True`.