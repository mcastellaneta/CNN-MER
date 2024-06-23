from importlib import reload


def reload_module(module):
    try:
        name = reload(module)
    except Exception:
        print(f'Error: module {module} is not defined')
        return

    print(f'Module {name} reloaded')