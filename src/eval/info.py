from . import constants as C


def _list_options(object_name: str):
    options = eval('C.{}.get_options()'.format(object_name.upper()))
    print_str = '{}:\n  - '.format(object_name) + '\n  - '.join(options)
    print(print_str)



def list_methods():
    _list_options('methods')

def list_metrics():
    _list_options('metrics')

def list_pp():
    _list_options('preprocess')

def list_workflows():
    _list_options('workflows')
