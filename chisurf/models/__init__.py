import chisurf.models.model
import chisurf.models.parse
import chisurf.models.fcs
import chisurf.models.tcspc
import chisurf.models.pda
import chisurf.models.global_model
import chisurf.models.stopped_flow
import chisurf.models.parameter_transform

from . model import *
from chisurf import logging


def function_to_model_decorator(**kws):

    def decorator(func):

        class ModelDecorator(chisurf.models.Model):

            def __init__(self, *args, **kwargs):
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
                ports = self._node.get_ports()
                logging.debug(f'ports: {ports}')
                logging.debug(f'ports.keys(): {ports.keys()}')
                logging.debug(f'ports.values(): {ports.values()}')
                for port_key in ports:
                    logging.debug(f'port_key: {port_key}')
                    port = ports[port_key]
                    logging.debug(f'port: {port}')
                    p = chisurf.fitting.parameter.FittingParameter(port=port, name=port_key)
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
                logging.debug(f'update_model called.')
                logging.debug(f'evaluating')
                self._node.evaluate()
                logging.debug(f'evaluate finished.')

            def update(self, **kwargs) -> None:
                logging.debug(f'update called.')
                logging.debug(f'find_parameters')
                self.find_parameters()
                logging.debug(f'find_parameters finished.')
                logging.debug(f'update_model')
                self.update_model()
                logging.debug(f'update_model finished.')

        return ModelDecorator

    return decorator
