import sys
import os
import webbrowser
from pathlib import Path
from jina import Document, Flow


def config():
    os.environ.setdefault('JINA_USE_CUDA', 'False')
    os.environ.setdefault('JINA_PORT_EXPOSE', '8886')
    os.environ.setdefault('JINA_WORKSPACE', './workspace')


def index(file_name):
    def load_marco(fn):
        cnt = 0
        with open(fn, 'r') as f:
            for ln, line in enumerate(f):
                try:
                    title, para = line.strip().split('\t')
                    doc = Document(
                        id=f'{cnt}',
                        uri=fn,
                        tags={'title': title, 'para': para})
                    cnt += 1
                    yield doc
                except:
                    print(f'skip line {ln}')
                    continue

    f = Flow().load_config('flows/index.yml')
    with f:
        f.post(on='/index', inputs=load_marco(file_name), show_progress=True, request_size=32)


def fillin_html():
    source_fn = Path(__file__).parent.absolute() / 'static/index_template.html'
    target_fn = Path(__file__).parent.absolute() / 'static/index.html'

    with open(source_fn, 'r') as fp, open(target_fn, 'w') as fw:
        t = fp.read()
        t = t.replace('{% JINA_PORT_EXPOSE %}',
                      f'{os.environ.get("JINA_PORT_EXPOSE")}')
        fw.write(t)


def query():
    from distutils.dir_util import copy_tree
    fillin_html()
    copy_tree('static', 'workspace/static')
    url_html_fn = Path(__file__).parent.absolute() / 'workspace/static/index.html'
    url_html_path = f'file://{url_html_fn}'
    f = Flow().load_config('flows/query.yml')
    with f:
        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass
        finally:
            print(f'You should see a demo page opened in your browser'
                  f'if not, you may open {url_html_path} manually')
        f.block()


def query_cli():
    def print_topk(resp):
        for doc in resp.docs:
            print(doc)
            doc = Document(doc)
            print(f'🤖 Answers:')
            for m in doc.matches:
                print(f'\t{m.tags["title"]}')
                print(f'\t{m.tags["para"]}')
                print(f'-----')

    f = Flow().load_config('flows/query.yml')
    with f:
        f.protocol = 'grpc'
        print(f'🤖 Hi there, please ask me questions related to the indexed Documents.\n'
              'For example, "Who is Paula Deen\'s brother?"\n')
        while True:
            text = input('Question: (type `\q` to quit)')
            if text == '\q' or not text:
                return
            f.post(on='/search', inputs=[Document(content=text), ], on_done=print_topk)


def main(task):
    config()
    if task == 'index':
        if Path('./workspace').exists():
            print('./workspace exists, please deleted it if you want to reindexi')
        data_fn = sys.argv[2] if len(sys.argv) >= 3 else 'toy_data/test.tsv'
        print(f'indexing {data_fn}')
        index(data_fn)
    elif task == 'query':
        query()
    elif task == 'query_cli':
        query_cli()


if __name__ == '__main__':
    task = sys.argv[1]
    main(task)
