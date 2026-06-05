from . model import *
from chisurf import logging
from chisurf.core.models.model import Model
from chisurf.core.fitting.parameter import FittingParameter


def function_to_model_decorator(**kws):
    """Create a decorator that wraps a callable into a `Model` subclass.

    The returned decorator turns a Python callable into a
    :class:`chisurf.core.models.Model` subclass that is backed by a ``chinet``
    node. Keyword arguments passed to this factory are forwarded to the
    model constructor.

    Parameters
    ----------
    **kws
        Keyword arguments forwarded to :class:`chisurf.core.models.Model` when the
        generated class is instantiated.

    Returns
    -------
    callable
        A decorator. When applied to a function it returns a new
        :class:`chisurf.core.models.Model` subclass.

    Examples
    --------
    Create a model class from a simple callback function. The resulting
    class can later be instantiated by the fitting framework.

    >>> def callback():
    ...     pass
    >>> ModelClass = function_to_model_decorator()(callback)
    >>> isinstance(ModelClass, type)
    True
    """

    def decorator(func):
        """Wrap ``func`` in a ``ModelDecorator`` class and return it.

        Parameters
        ----------
        func : callable
            The Python callable to wrap.

        Returns
        -------
        ModelDecorator
            A dynamically created :class:`chisurf.core.models.Model` subclass.
        """

        class ModelDecorator(Model):

            def __init__(self, *args, **kwargs):
                """Initialize the model decorator with chinet node and parameters.

                Parameters
                ----------
                *args
                    Positional arguments forwarded to the parent Model.
                **kwargs
                    Keyword arguments forwarded to the parent Model, updated
                    with the factory-level keyword arguments from
                    :func:`function_to_model_decorator`.
                """
                logging.info('ModelDecorator.__init__')
                logging.debug(f'args: {args}')
                logging.debug(f'kwargs: {kwargs}')
                logging.debug(f'kws: {kws}')
                logging.debug(f'updating kwargs with kws')
                kwargs.update(kws)
                logging.debug(f'updating kwargs finished')
                logging.debug(f'importing chinet')
                import chinet as cn
                logging.debug(f'importing chinet finished')
                logging.debug(f'chinet: {cn}')
                super(ModelDecorator, self).__init__(*args, **kwargs)
                logging.debug(f'super called.')
                self._node = cn.Node()
                logging.debug(f'_node: {self._node}')
                self._node.set_python_callback_function(func)
                logging.debug(f'_node.set_python_callback_function called.')
                logging.debug(f'func: {func}')
                self.node_parameters = list()
                logging.debug(f'node_parameters: {self.node_parameters}')
                self.make_parameters()
                logging.debug(f'make_parameters finished.')

            def make_parameters(self):
                """Create FittingParameters from the chinet node ports."""
                ports = self._node.get_ports()
                logging.debug(f'ports: {ports}')
                logging.debug(f'ports.keys(): {ports.keys()}')
                logging.debug(f'ports.values(): {ports.values()}')
                for port_key in ports:
                    logging.debug(f'port_key: {port_key}')
                    port = ports[port_key]
                    logging.debug(f'port: {port}')
                    p = FittingParameter(port=port, name=port_key)
                    logging.debug(f'p: {p}')
                    logging.debug(f'p.name: {p.name}')
                    self.node_parameters.append(p)
                    logging.debug(f'node_parameters: {self.node_parameters}')
                self.find_parameters()
                logging.debug(f'find_parameters finished.')

                # output ports act as fixed parameters
                logging.debug(f'outputs: {self._node.outputs}')
                logging.debug(f'fixing output ports')
                for port_key in self._node.outputs:
                    logging.debug(f'port_key: {port_key}')
                    self.parameters_all_dict[port_key].fixed = True
                    logging.debug(f'fixed: {self.parameters_all_dict[port_key].fixed}')
                logging.debug(f'fixed output ports finished.')

            def update_model(self, **kwargs):
                """Evaluate the chinet node to compute the model output."""
                logging.debug(f'update_model called.')
                logging.debug(f'evaluating')
                self._node.evaluate()
                logging.debug(f'evaluate finished.')

            def update(self, **kwargs) -> None:
                """Refresh parameters and re-evaluate the model."""
                logging.debug(f'update called.')
                logging.debug(f'find_parameters')
                self.find_parameters()
                logging.debug(f'find_parameters finished.')
                logging.debug(f'update_model')
                self.update_model()
                logging.debug(f'update_model finished.')

        return ModelDecorator

    return decorator
