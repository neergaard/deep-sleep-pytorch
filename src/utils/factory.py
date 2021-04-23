import importlib


def create_instance(config_map):
    '''Expects a string that can be imported as with a module.class name'''
    module_name, class_name = config_map['import'].rsplit(".", 1)

    try:
        print('Importing {}.{}'.format(module_name, class_name))
        somemodule = importlib.import_module(module_name)
        # print('getattr '+class_name)
        cls_instance = getattr(somemodule, class_name)
        # print(cls_instance)
    except Exception as err:
        print("Creating error: {0}".format(err))
        exit(-1)

    return cls_instance
