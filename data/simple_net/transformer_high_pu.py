import numpy as np
import pandapower as pp


class TrafoHighPuNet:
    def __init__(self, vm_pu, p_mw, q_mvar, vn_lv_kv, init_vm_pu):
        self.vm_pu = vm_pu
        self.p_mw = p_mw
        self.q_mvar = q_mvar
        self.vn_lv_kv = vn_lv_kv
        self.init_vm_pu = init_vm_pu
        self.net = pp.create_empty_network()
        self._create_network()

    def _create_network(self):
        pp.create_bus(self.net, vn_kv=10.0, index=0)
        pp.create_bus(self.net, vn_kv=10.0, index=1)
        pp.create_transformer_from_parameters(
            self.net,
            hv_bus=0,
            lv_bus=1,
            sn_mva=100.0,
            vk_percent=10,
            vkr_percent=0.5,
            vn_hv_kv=10.0,
            vn_lv_kv=self.vn_lv_kv,
            pfe_kw=0.0,
            i0_percent=0.5,
            phase_shift=0.0,
        )
        pp.create_ext_grid(self.net, bus=0, vm_pu=self.vm_pu, va_degree=0.0)
        pp.create_load(self.net, bus=1, p_mw=self.p_mw, q_mvar=self.q_mvar)

    def run_power_flow(self, init_vm_pu=None):
        if init_vm_pu is None:
            init_vm_pu = self.init_vm_pu
        try:
            pp.runpp(self.net, algorithm="nr", init="auto", init_vm_pu=init_vm_pu, init_va_degree=[0.0, 0.0])
        except pp.LoadflowNotConverged as e:
            print("Initial value can not converge")
            print(e)
        else:
            print("Initial value converged")
            print(self.net.res_bus)
            print(self.net.res_trafo)


if __name__ == "__main__":
    # value range arbitrary
    net_gen = TrafoHighPuNet(vm_pu=1.0, p_mw=0.0, q_mvar=0.0, vn_lv_kv=25.0, init_vm_pu=[1.0, 1.0])
    net_gen.run_power_flow()
    net_gen.run_power_flow(init_vm_pu=[1.0, 1.5])

    for _ in range(10):
        vm_pu = np.random.uniform(0.9, 3.0)
        p_mw = np.random.uniform(0.0, 1000.0)
        q_mvar = np.random.uniform(0.0, 0.5)
        vn_lv_kv = np.random.uniform(0.9, 3.0)
        init_vm_pu = [np.random.uniform(0.9, 3.0), np.random.uniform(0.9, 3.0)]

        net_instance = TrafoHighPuNet(vm_pu, p_mw, q_mvar, vn_lv_kv, init_vm_pu)
        print(f"Run {_+1}:")
        net_instance.run_power_flow()
        net_instance.run_power_flow([1.0, 2.5])
