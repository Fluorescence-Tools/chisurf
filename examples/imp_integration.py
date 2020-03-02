import IMP
import IMP.core


class IMPDecorator(IMP.Decorator):

    def __init__(self):
        super().__init__()

    def do_setup_particle(
            self,
            m: IMP.Model,
            pi: IMP.ParticleIndex,
            value: float
    ):
        pass

    def get_is_setup(self):
        pass

    def get_decorator_name(self):
        pass

    def set_decorator_name(self):
        pass

    def show(self):
        pass
