
class StorageInfo:
    pass


class InputStorage(StorageInfo):
    def __init__(self, *ids):
        self.ids = ids


class OutputStorage(StorageInfo):
    pass


class IntermediateStorage(StorageInfo):
    def __init__(self, size_fn):
        self.size = size_fn


class OP:
    names = []
    backward_storage: StorageInfo = None

    def intermediate(self):
        return None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


registry = {}
all_modes = ["none", "normal", "multiway", "newnode", "multiway_newnode", "conv_multiway_newnode", "conv_normal"]

def implements(names, modes):
    def _wrapper(C):
        for m in modes:
            assert m in all_modes
            if m not in registry:
                registry[m] = {}
            for n in names:
                if n not in registry[m]:
                    registry[m][n] = []
                registry[m][n].append(C)
        C.names = names
        return C
    return _wrapper


def list_ops(mode, name):
    return registry[mode][name]
