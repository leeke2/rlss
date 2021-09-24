class BaseClass:
    def __init__(self, **kwargs):

        if '_REQUIRED_PARAMS' not in self.__dict__:
            params = ['device']
        else:
            params = self._REQUIRED_PARAMS + ['device']

        parameters = {
            param: kwargs.get(param)
            for param in params
        }

        self.__dict__.update(parameters)
