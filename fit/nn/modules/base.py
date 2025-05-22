class Layer:
    def __init__(self):
        self._parameters = []
        self._children = []

    def parameters(self):
        params = list(self._parameters)
        for child in self._children:
            params += child.parameters()
        return params

    def add_parameter(self, param):
        self._parameters.append(param)

    def add_child(self, layer):
        self._children.append(layer)

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
