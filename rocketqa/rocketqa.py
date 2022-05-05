import os
import sys
import json
import paddle
import urllib
import numpy as np
import tarfile
import warnings
import hashlib
from tqdm import tqdm
from rocketqa.encoder.dual_encoder import DualEncoder
from rocketqa.encoder.cross_encoder import CrossEncoder

paddle.enable_static()
warnings.simplefilter('ignore')

__MODELS = {
        "v1_marco_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_marco_de.tar.gz",       # RocketQA v1 dual-encoder trained on MSMARCO
        "v1_marco_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_marco_ce.tar.gz",       # RocketQA v1 cross-encoder trained on MSMARCO
        "v1_nq_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_nq_de.tar.gz",             # RocketQA v1 dual-encoder trained on Natural Question
        "v1_nq_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/v1_nq_ce.tar.gz",             # RocketQA v1 cross-encoder trained on Natural Question
        "pair_marco_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/pair_marco_de.tar.gz",   # PAIR dual-encoder trained on MSMARCO
        "pair_nq_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/pair_nq_de.tar.gz",         # PAIR dual-encoder trained on Natural Question
        "v2_marco_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v2_marco_de.tar.gz",       # RocketQA v2 dual-encoder trained on MSMARCO
        "v2_marco_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/v2_marco_ce.tar.gz",       # RocketQA v2 cross-encoder trained on MSMARCO
        "v2_nq_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/v2_nq_de.tar.gz",             # RocketQA v2 dual-encoder trained on Natural Question
        "zh_dureader_de": "http://rocketqa.bj.bcebos.com/RocketQAModels/zh_dureader_de.tar.gz", # RocketQA zh dual-encoder trained on Dureader
        "zh_dureader_ce": "http://rocketqa.bj.bcebos.com/RocketQAModels/zh_dureader_ce.tar.gz", # RocketQA zh cross-encoder trained on Dureader
        "zh_dureader_de_v2": "http://rocketqa.bj.bcebos.com/RocketQAModels/zh_dureader_de_v2.tar.gz",
        "zh_dureader_ce_v2": "http://rocketqa.bj.bcebos.com/RocketQAModels/zh_dureader_ce_v2.tar.gz"
}

__MODELS_MD5 = {
        "v1_marco_de": "d8210e4080935bd7fdad7a394cd60b66",
        "v1_marco_ce": "caec5aedc46f22edd7107ecd793fc7fb",
        "v1_nq_de": "cfeb70f82087b8a47bb0d6d6cfcd61c5",
        "v1_nq_ce": "15aac78d70cc25994016b8a30d80f12c",
        "pair_marco_de": "b4080ffa2999525e5ba2aa1f4e03a9e8",
        "pair_nq_de": "d770bc379ec6def7e0588ec02c80ace2",
        "v2_marco_de": "4ce64ff35d1d831f0ca989e49abde227",
        "v2_marco_ce": "915ea7ff214a4a92a3a1e1d56c3fb469",
        "v2_nq_de": "8f177aa75cadaad6656dcd981edc983b",
        "zh_dureader_de": "39811675289c311236c667ad57ebd2d2",
        "zh_dureader_ce": "11caeb179febc5f0a55fa10ae3f2d123",
        "zh_dureader_de_v2": "889e62b0091bc350622549b57a2616ec",
        "zh_dureader_ce_v2": "552675c98c546e798a33cc84325921f6"
}

def available_models():
    """
    Return the names of available RocketQA models
    """
    return __MODELS.keys()


def load_model(model, use_cuda=False, device_id=0, batch_size=1):
    """
    Load a RocketQA model or an user-specified checkpoint
    Args:
        model: A model name return by `rocketqa.available_models()` or the path of an user-specified checkpoint config
        use_cuda: Whether to use GPU
        device_id: The device to put the model
        batch_size: Batch_size during inference
    Returns:
        model
    """

    model_type = ''
    model_name = ''
    rocketqa_model = False
    encoder_conf = {}

    if model in __MODELS:
        model_name = model
        print (f"RocketQA model [{model_name}]", file=sys.stderr)
        rocketqa_model = True
        model_path = os.path.expanduser('~/.rocketqa/') + model_name + '/'
        if not os.path.exists(model_path):
            if __download(model_name) is False:
                raise Exception(f"RocketQA model [{model_name}] download failed, \
                        please check model dir [{model_path}]")

        encoder_conf['conf_path'] = model_path + 'config.json'
        encoder_conf['model_path'] = model_path
        if model_name.find("_de") >= 0:
            model_type = 'dual_encoder'
        elif model_name.find("_ce") >= 0:
            model_type = 'cross_encoder'

    if rocketqa_model is False:
        print ("User-specified model", file=sys.stderr)
        conf_path = model
        model_name = model
        if not os.path.isfile(conf_path):
            raise Exception(f"Config file [{conf_path}] not found")
        try:
            with open(conf_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception as e:
            raise Exception(str(e) + f"\nConfig file [{conf_path}] load failed")

        encoder_conf['conf_path'] = conf_path

        split_p = conf_path.rfind('/')
        if split_p > 0:
            encoder_conf['model_path'] = conf_path[0:split_p + 1]

        if "model_type" not in config_dict:
            raise Exception("[model_type] not found in config file")
        model_type = config_dict["model_type"]
        if model_type != "dual_encoder" and model_type != "cross_encoder":
            raise Exception("model_type [model_type] is illegal, must be `dual_encoder` or `cross_encoder`")

    encoder_conf["use_cuda"] = use_cuda
    encoder_conf["device_id"] = device_id
    encoder_conf["batch_size"] = batch_size
    encoder_conf["model_name"] = model_name

    if model_type[0] == "d":
        encoder = DualEncoder(**encoder_conf)
    elif model_type[0] == "c":
        encoder = CrossEncoder(**encoder_conf)
    print ("Load model done", file=sys.stderr)
    return encoder


def __download(model_name):
    os.makedirs(os.path.expanduser('~/.rocketqa/'), exist_ok=True)
    filename = model_name + '.tar.gz'
    download_dst = os.path.join(os.path.expanduser('~/.rocketqa/') + filename)
    download_url = __MODELS[model_name]

    if not os.path.exists(download_dst):
        print (f"Download RocketQA model [{model_name}]", file=sys.stderr)
        with urllib.request.urlopen(download_url) as source, open(download_dst, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

    file_md5= __get_file_md5(download_dst)
    if file_md5 != __MODELS_MD5[model_name]:
        raise Exception(f"Model file [{download_dst}] exists, but md5 doesnot match")

    try:
        t = tarfile.open(download_dst)
        t.extractall(os.path.expanduser('~/.rocketqa/'))
    except Exception as e:
        print (str(e), file=sys.stderr)
        return False

    return True

def __get_file_md5(fname):
    m = hashlib.md5()
    with open(fname,'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)

    return m.hexdigest()


if __name__ == '__main__':
    pass
