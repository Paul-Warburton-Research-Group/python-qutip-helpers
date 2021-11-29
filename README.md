## Python Helper Codes for Small-Scale Simulations

#### `annealing.py`

Provides the class `IsingGraph` which is used to initialise [`qutip`](https://github.com/qutip) Two-Level-System (TLS) Hamiltonians for Quantum Annealing (QA), from a [`networkx`](https://github.com/networkx) undirected graph.

To create arbitrary TLS Hamiltonians with coupling elements of any order, a graph associated with each order of the couplings can be passed using `setNthOrderGraph()`:
 - 1-local terms are initialised based on the nodes of the graph.
 - 2-local terms are initialised based on the edges of the graph.
 - 3-local terms are initialised based on a loop formed by three edges connecting three nodes.
 - 4-local terms are initialised based on a loop formed by four edges connecting four nodes.
 - etc...

When an `IsingGraph` is initialised using these graphs, terms such as `hz`, `Jxy`, `Jxzy`, etc.. can be assigned values and schedules using `setCoef()`. Schedules are provided as callback functions with signature `def my_schedule(s, p)` where `s` is dimensionless time and `p` a dictionary of parameters used in the schedule.

The `qutip` Hamiltonian at a particular value of `s` can be obtained using `getHamiltonian()`. For dynamics simulations within `qutip`, use `getQobjEvo()` and pass the `QobjEvo` directly to solvers such as `sesolve()` and `mesolve()`.

There are also many other functions and graph plotting functionality for convenience.
