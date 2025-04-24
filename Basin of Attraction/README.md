# <span style="color:red"> Analytical Method</span>
## *Estimating the Basin of Attraction*

---
---

## **Overview**

This code is to estimate the basin of attraction of any grid with any numbers of buses including meshed or radial.

---

### **Required libraries:**

- NumPy: pip install numpy
- Matplotlib: pip install matplotlib
- Pandapower: pip install pandapower

---

#### ***Estimating_the_Basin_of_Attraction.ipynb***

The provided code is written in Python and can be run for any grid. To change the grid, the *net* parameter need to be defined based on the desired network with the standard defined in pandapower library.

The *v* parameter can be selected as slack bus voltage, the nominal voltages of the grid, or the voltages calculated in the privious run of power flow calculations.

Other parameters can be easily extracted from the network.

#### ***Results__Estimating_the_Basin_of_Attraction__7_Bus.png***

One example of minimum and maximum radius of the basin of attraction for 7-bus system.

---
---
