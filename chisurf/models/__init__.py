import chisurf.models.model
import chisurf.models.parse
import chisurf.models.fcs
import chisurf.models.tcspc
import chisurf.models.pda
import chisurf.models.global_model
import chisurf.models.stopped_flow
from . model import *


def function_to_model_decorator(**kws):

    def decorator(func):
        class ModelDecorator(chisurf.models.Model):
            def __init__(self, *args, **kwargs):
                kwargs.update(kws)
                import chinet as cn
                super(ModelDecorator, self).__init__(*args, **kwargs)
                self._node = cn.Node()
                self._node.set_python_callback_function(func)
                self.node_parameters = list()
                for p in self.node_parameters:
                    p.reactive = True
                self.make_parameters()

            def make_parameters(self):
                ports = self._node.get_ports()
                for port_key in ports:
                    port = ports[port_key]
                    p = chisurf.fitting.parameter.FittingParameter(port=port)
                    self.node_parameters.append(p)
                self.find_parameters()
                # output ports act as fixed parameters
                for port_key in self._node.outputs:
                    self.parameter_dict[port_key].fixed = True

            def update_model(self, **kwargs):
                self._node.evaluate()

            def update(self, **kwargs) -> None:
                self.find_parameters()
                self.update_model()

        return ModelDecorator

    return decorator

