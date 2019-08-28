from __future__ import annotations

import json
import re
from typing import ValuesView

import yaml
from slugify import slugify

import mfm


class Base(object):

    @property
    def name(self) -> str:
        try:
            name = self._kw['name']
            return name() if callable(name) else name
        except KeyError or AttributeError:
            return self.__class__.__name__

    @name.setter
    def name(
            self,
            v: str
    ):
        self._kw['name'] = v

    def save(
            self,
            filename: str,
            file_type: str = 'json'
    ) -> None:
        if file_type == "yaml":
            txt = self.to_yaml()
        else:
            txt = self.to_json()
        with open(filename, 'w') as fp:
            fp.write(txt)

    def load(
            self,
            filename: str,
            file_type: str = 'json'
    ) -> None:
        if file_type == "json":
            self.from_json(filename=filename)
        else:
            self.from_yaml(filename=filename)

    def to_dict(self) -> dict:
        try:
            return self._kw
        except AttributeError:
            self._kw = dict()
            return self._kw

    def from_dict(
            self,
            v: dict
    ) -> None:
        self._kw = v

    def to_json(
            self,
            indent: int = 4,
            sort_keys: bool = True
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=sort_keys
        )

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())

    def from_yaml(
            self,
            yaml_string: str = None,
            filename: str = None
    ) -> None:
        if filename is not None:
            with open(filename, 'r') as fp:
                j = yaml.load(fp)
        elif yaml_string is not None:
            j = json.loads(yaml_string)
        else:
            j = dict()
        self.from_dict(j)

    def from_json(
            self,
            json_string: str = None,
            filename: str = None
    ) -> None:
        """Reads the content of a JSON file into the object.

        Example
        -------

        >>> import mfm
        >>> dc = mfm.curve.DataCurve()
        >>> dc.from_json(filename='./sample_data/internal_types/datacurve.json')


        Parameters
        ----------
        json_string : str
            A string containing the JSON file
        filename: str
            The filename to be opened

        """
        j = dict()
        if filename is not None:
            with open(filename, 'r') as fp:
                j = json.load(fp)
        elif json_string is not None:
            j = json.loads(json_string)
        else:
            pass
        self.from_dict(j)

    def __setattr__(
            self,
            k: str,
            v: object
    ):
        try:
            kw = self.__dict__['_kw']
        except KeyError:
            kw = dict()
            self.__dict__['_kw'] = kw
        if k in kw:
            self.__dict__['_kw'][k] = v

        # Set the attributes normally
        propobj = getattr(self.__class__, k, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, v)
        else:
            super(Base, self).__setattr__(k, v)

    def __getattr__(
            self,
            key: str
    ):
        return self._kw.get(key, None)

    def __str__(self):
        s = 'class: %s\n' % self.__class__.__name__
        s += self.to_yaml()
        return s

    def __init__(self, *args, **kwargs):
        """The class saves all passed keyword arguments in dictionary and makes
        these keywords accessible as attributes. Moreover, this class may saves these
        keywords in a JSON or YAML file. These files can be also loaded.

        :param name:
        :param args:
        :param kwargs:

        Example
        -------

        >>> import mfm
        >>> bc = mfm.Base(parameter="ala", lol=1)
        >>> bc.lol
        1
        >>> bc.parameter
        ala
        >>> bc.to_dict()
        {'lol': 1, 'parameter': 'ala', 'verbose': False}
        >>> bc.from_dict({'jj': 22, 'zu': "auf"})
        >>> bc.jj
        22
        >>> bc.zu
        auf
        """
        super(Base, self).__init__()
        if len(args) > 0 and isinstance(args[0], dict):
            kwargs = args[0]

        # clean up the keys (no spaces etc)
        d = dict()
        regex_pattern = r'[^-a-z0-9_]+'
        for key in kwargs:
            r = slugify(key, separator='_', regex_pattern=regex_pattern)
            d[r] = kwargs[key]

        # Assign the the names and set standard values
        name = kwargs.pop('name', self.__class__.__name__)
        d['name'] = name() if callable(name) else name
        d['verbose'] = d.get('verbose', False)
        self._kw = dict()
        self._kw = d


def clean_string(
        s: str
) -> str:
    """ Remove special characters to clean up string and make it compatible
    with a Python variable names

    :param s:
    :return:
    """
    s = re.sub('[^0-9a-zA-Z_]', '', s)
    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)
    return s


def find_objects(
        search_list: ValuesView,
        object_type,
        remove_double: bool = True):
    """Traverse a list recursively a an return all objects of type `object_type` as
    a list

    :param search_list: list
    :param object_type: an object type
    :param remove_double: boolean
    :return: list of objects with certain object type
    """
    re = []
    for value in search_list:
        if isinstance(value, object_type):
            re.append(value)
        elif isinstance(value, list):
            re += find_objects(value, object_type)
    if remove_double:
        return list(set(re))
    else:
        return re