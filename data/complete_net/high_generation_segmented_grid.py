# source(10kV)-node_0-line(1 ohm)-node_1-load(25*-15 = -375MW)
# node_0 = 10kV
# node_1 = 25kV (should be the answer), node_1 = -15kV (V = 15kV, angle = 180 degrees)

import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged
import numpy as np

N_SEGMENTS = 10

class HighGenSegmentedNet:
    def __init__(self, vm_pu, p_mw, q_mvar, total_p_gen, total_q_gen, init_vm_pu):
        self.vm_pu = vm_pu
        self.p_mw = p_mw
        self.q_mvar = q_mvar
        self.total_p_gen = total_p_gen
        self.total_q_gen = total_q_gen
        self.init_vm_pu = init_vm_pu
        self.net = pp.create_empty_network()
        self._create_network()
    
    def _create_network(self):
        exp = np.exp(np.linspace(0, 1, N_SEGMENTS))

        p_gens = exp / exp.sum() * self.total_p_gen
        q_gens = exp / exp.sum() * self.total_q_gen
        resistances = [1.0 / N_SEGMENTS] * N_SEGMENTS
        
        pp.create_bus(self.net, vn_kv=10.0, index=0)

        from_bus = 0
        for segment, resistance, segment_p_gen, segment_q_gen in zip(range(N_SEGMENTS), resistances, p_gens, q_gens):
            to_bus = pp.create_bus(self.net, vn_kv=10.0)
            pp.create_line_from_parameters(
                self.net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=1.0,
                r_ohm_per_km=resistance,
                x_ohm_per_km=0.00001,
                c_nf_per_km=0.0,
                g_us_per_km=0.0,
                max_i_ka=100.0
            )
            pp.create_sgen(self.net, bus=to_bus, p_mw=segment_p_gen, q_mvar=segment_q_gen)
            from_bus = to_bus

        pp.create_sgen(self.net, bus=to_bus, p_mw=self.p_mw, q_mvar=170.0)  # parametric p_mw, q_mvar
        pp.create_ext_grid(self.net, bus=0, vm_pu=self.vm_pu, va_degree=0.0)  # parametrize vm_pu
    
    def run_power_flow(self, init_vm_pu=None):
        if init_vm_pu is None:
            init_vm_pu = self.init_vm_pu
        try:
            pp.runpp(self.net, algorithm="nr", init="auto", init_vm_pu=init_vm_pu, init_va_degree=0.0)
        except LoadflowNotConverged as e:
            print("Initial value can not converge")
            print(e)
        else:
            print("Initial value converged")
            print(self.net.res_bus)
            print(self.net.res_trafo)

'''
succeed with init_vm_pu=auto, result vm_pu bus 1: 0.8772050081247648
Wrong initial value for init_vm_pu=0.5. Error:Power Flow nr did not converge after 10 iterations!
succeed with init_vm_pu=1.0, result vm_pu bus 1: 0.8772050081247648
succeed with init_vm_pu=2.0, result vm_pu bus 1: 1.1431280911036827
succeed with init_vm_pu=3.0, result vm_pu bus 1: 1.1431280911036816
succeed with init_vm_pu=4.0, result vm_pu bus 1: 1.1431280911037756
'''

if __name__ == "__main__":
    net_instance = HighGenSegmentedNet(vm_pu=1.0, total_p_gen=375.0, total_q_gen=0.0, p_mw=0.0, q_mvar=170.0, init_vm_pu="auto")
    net_instance.run_power_flow()
    for init_vm_pu in ["auto", 0.5, 1.0, 2.0, 3.0, 4.0]:
        net_instance.run_power_flow(init_vm_pu=init_vm_pu)
