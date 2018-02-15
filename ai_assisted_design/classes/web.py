from yattag import Doc, indent
import os


def save_html(html_string):

    _path = '{0}/renderer/index.html'.format(os.getcwd())

    with open(_path, 'w') as html_file:
        html_file.write(html_string)
        html_file.close()


def create_template(template_title, injected_content):
    doc, tag, text = Doc().tagtext()

    _content = ''

    for item in injected_content:
        _content += '<{0}></{0}>'.format(item)

    with tag('html'):
        with tag('head'):
            with tag('title'):
                text(template_title)
            doc.asis('<link rel="stylesheet" href="./css/style.css">')
        with tag('body'):
            doc.asis(_content)

    print(indent(doc.getvalue()))
    print('-'*20)

    save_html(indent(doc.getvalue()))


def clear():
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        with tag('head'):
            with tag('title'):
                text('Waiting for input...')
            doc.asis('<link rel="stylesheet" href="./css/style.css">')
        with tag('body'):
            text('Waiting for input...')

    print(indent(doc.getvalue()))
    print('-' * 20)

    save_html(indent(doc.getvalue()))
