# source(10kV)-node_0-line(1 ohm)-node_1-load(25*-15 = -375MW)
# node_0 = 10kV
# node_1 = 25kV (should be the answer), node_1 = -15kV (V = 15kV, angle = 180 degrees)

import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged
import numpy as np

n_segments = 10
total_gen = 25.0 * 15.0

exp = np.exp(np.linspace(0, 1, n_segments))
exp_reversed = np.exp(np.linspace(0, 1, n_segments))[::-1]
linear = np.linspace(0, 1, n_segments)
uniform = np.array([1.0] * n_segments)
step=9
step_fn = np.array([0.0] * step + [1.0] * (n_segments - step))

gens = exp / exp.sum() * total_gen
resistances = [1.0 / n_segments] * n_segments
# resistances = step_fn / step_fn.sum() * 1.0

def create_pp_segment(net, from_bus, to_bus, resistance):
    pp.create_line_from_parameters(
        net,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=1.0,
        r_ohm_per_km=resistance,
        x_ohm_per_km=0.00001,
        c_nf_per_km=0.0,
        g_us_per_km=0.0,
        max_i_ka=100.0
    )
    return net


net = pp.create_empty_network()
pp.create_bus(net, vn_kv=10.0, index=0)
pp.create_bus(net, vn_kv=10.0, index=1)
pp.create_transformer_from_parameters(
    net,
    hv_bus=0,
    lv_bus=1,
    sn_mva=100.0,
    vk_percent=10,
    vkr_percent=0.5,
    vn_hv_kv=10.0,
    vn_lv_kv=25.0,  # parametrize vn_lv_kv
    pfe_kw=0.0,
    i0_percent=0.5,
    phase_shift=0.0,
)
from_bus = 1
for segment, resistance, segment_gen in zip(range(n_segments), resistances, gens):
    to_bus = pp.create_bus(net, vn_kv=10.0)
    create_pp_segment(net, from_bus=from_bus, to_bus=to_bus, resistance=resistance)
    pp.create_sgen(net, bus=to_bus, p_mw=segment_gen, q_mvar=0.0)  # parametric p_mw, q_mvar
    from_bus = to_bus


pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0)  # parametrize vm_pu

for init_vm_pu in ["auto", 0.5, 1.0, 1.5, 2.0, 2.5]:
    try:
        pp.runpp(net, algorithm="nr", init_vm_pu=init_vm_pu, init_va_degree=0.0)  # wrong initial value, parametrize init_vm_pu
        print(f"succeed with init_vm_pu={init_vm_pu}")
    except LoadflowNotConverged as e:
        print(f"Wrong initial value for init_vm_pu={init_vm_pu}. Error:{e}")

print("result vm_pu")
print(net.res_bus)