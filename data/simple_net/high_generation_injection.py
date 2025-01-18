# Base network with high generation injection, used parameterization
# source(10kV)-node_0-line(1 ohm)-node_1-load(25*-15 = -375MW)
# node_0 = 10kV
# node_1 = 25kV (should be the answer), node_1 = -15kV (V = 15kV, angle = 180 degrees)

import numpy as np
import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged


class HighGenInjectionNet:
    def __init__(self, vm_pu, p_mw, q_mvar, init_vm_pu):
        self.vm_pu = vm_pu
        self.p_mw = p_mw
        self.q_mvar = q_mvar
        self.init_vm_pu = init_vm_pu
        self.net = pp.create_empty_network()
        self._create_network()

    def _create_network(self):
        pp.create_bus(self.net, vn_kv=10.0, index=0)
        pp.create_bus(self.net, vn_kv=10.0, index=1)
        pp.create_line_from_parameters(
            self.net,
            from_bus=0,
            to_bus=1,
            length_km=1.0,
            r_ohm_per_km=1.0,
            x_ohm_per_km=0.0,
            c_nf_per_km=0.0,
            g_us_per_km=0.0,
            max_i_ka=100.0,
        )
        pp.create_ext_grid(self.net, bus=0, vm_pu=self.vm_pu, va_degree=0.0)
        pp.create_sgen(self.net, bus=1, p_mw=self.p_mw, q_mvar=self.q_mvar)

    def run_power_flow(self, init_vm_pu=None):
        if init_vm_pu is None:
            init_vm_pu = self.init_vm_pu
        try:
            pp.runpp(self.net, algorithm="nr", init="auto", init_vm_pu=init_vm_pu, init_va_degree=[0.0, 0.0])
        except LoadflowNotConverged as e:
            print("Initial value can not converge")
            print(e)
        else:
            print("Initial value converged")
            print(self.net.res_bus)
            print(self.net.res_trafo)


if __name__ == "__main__":
    # value range arbitrary
    net_gen = HighGenInjectionNet(vm_pu=1.0, p_mw=25.0 * 15.0, q_mvar=0.0, init_vm_pu=[1.0, 1.0])
    net_gen.run_power_flow()
    net_gen.run_power_flow(init_vm_pu=[1.0, 2.5])

    for _ in range(10):
        vm_pu = np.random.uniform(0.9, 3.0)
        p_mw = np.random.uniform(0.0, 1000.0)
        q_mvar = np.random.uniform(0.0, 0.5)
        init_vm_pu = [np.random.uniform(0.9, 3.0), np.random.uniform(0.9, 3.0)]

        net_instance = HighGenInjectionNet(vm_pu, p_mw, q_mvar, init_vm_pu)
        print(f"Run {_+1}:")
        net_instance.run_power_flow([1.0, 2.5])
