import pandapower as pp


class SimpleTwoBus:
    def __init__(self, V_ext, P, Q, R, X):
        """This class creates a simple 2-bus network."""
        self.net = pp.create_empty_network()
        bus1 = pp.create_bus(self.net, vn_kv=1.0, name="Bus 1")
        bus2 = pp.create_bus(self.net, vn_kv=1.0, name="Bus 2")

        pp.create_line_from_parameters(
            self.net,
            from_bus=0,
            to_bus=1,
            length_km=1.0,
            r_ohm_per_km=R,
            x_ohm_per_km=X,
            c_nf_per_km=0.0,
            g_us_per_km=0.0,
            max_i_ka=100.0,
        )

        pp.create_load(self.net, bus2, p_mw=P, q_mvar=Q, name="Load")

        # Create an external grid connection at bus 1 with specified G and B
        pp.create_ext_grid(self.net, bus1, vm_pu=V_ext, name="Grid Connection")

        # self.create_two_bus_grid()

    def run_power_flow(self):
        pp.runpp(self.net)

        flat_start_solution_V = self.net.res_bus[["vm_pu"]].values
        flat_start_solution_angle = self.net.res_bus[["va_degree"]].values
        return flat_start_solution_V, flat_start_solution_angle

    # def create_two_bus_grid(self):
    #     # Create two buses with initialized voltage and angle

    #     # Initialize voltage and angle for buses
    #     self.net.bus.loc[bus1, "vm_pu"] = self.V_init[0]
    #     self.net.bus.loc[bus1, "va_degree"] = self.theta_init[0]
    #     self.net.bus.loc[bus2, "vm_pu"] = self.V_init[1]
    #     self.net.bus.loc[bus2, "va_degree"] = self.theta_init[1]

    #     # create a line between the two buses

    #     # Create a transformer between the two buses
    #     # pp.create_transformer(self.net, bus1, bus2, std_type="0.25 MVA 20/0.4 kV")

    #     # Create a load at bus 2 with specified P and Q
