
class EnvDict:

    _env_dict = {}

    @staticmethod
    def set_value(cls, key, value):
        cls._env_dict[key] = value

    @staticmethod
    def get_value(cls, key):
        return cls._env_dict[key]