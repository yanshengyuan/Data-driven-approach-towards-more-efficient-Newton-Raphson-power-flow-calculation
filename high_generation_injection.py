# source(10kV)-node_0-line(1 ohm)-node_1-load(25*-15 = -375MW)
# node_0 = 10kV
# node_1 = 25kV (should be the answer), node_1 = -15kV (V = 15kV, angle = 180 degrees)

import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged


net = pp.create_empty_network()
pp.create_bus(net, vn_kv=10.0, index=0)
pp.create_bus(net, vn_kv=10.0, index=1)
pp.create_line_from_parameters(
    net,
    from_bus=0,
    to_bus=1,
    length_km=1.0,
    r_ohm_per_km=1.0,
    x_ohm_per_km=0.0,
    c_nf_per_km=0.0,
    g_us_per_km=0.0,
    max_i_ka=100.0
)
pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0)  # parametrize vm_pu
pp.create_sgen(net, bus=1, p_mw=25.0 * 15.0, q_mvar=0.0)  # parametric p_mw, q_mvar

try:
    pp.runpp(net, algorithm="nr", init="auto", init_vm_pu=[1.0, 1.0], init_va_degree=[0.0, 0.0])  # wrong initial value, parametrize init_vm_pu
except LoadflowNotConverged as e:
    print("Wrong initial value")
    print(e)

pp.runpp(net, algorithm="nr", init="auto", init_vm_pu=[1.0, 2.5], init_va_degree=[0.0, 0.0])  # correct initial value, parametrize init_vm_pu
print("Correct initial value")
print(net.res_bus)
print(net.res_trafo)
