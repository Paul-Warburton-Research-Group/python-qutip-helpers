import time
import numpy as np
import networkx as nx
import qutip as qt
import itertools as itt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pycqed.util as util

class IsingGraph:
    """ Class to convert a series of graphs to an Ising Hamiltonian.
    """
    
    # Graph visual properties
    __node_color = 'C0'
    __node_size = 1500
    __edge_color = 'k'
    __edge_size = 2
    __label_font_size = 16
    __label_font_weight = "normal"
    __paulis = {
        "i": qt.qeye(2),
        "x": qt.sigmax(),
        "y": qt.sigmay(),
        "z": qt.sigmaz()
    }
    __node_labels = [
        "node_name",
        "paulis"
    ]
    
    def __init__(self, graph_family=None):
        
        self.graph_family = {}                          # The family of graphs
        self.coefs_of_order = {}                        # The data structure used to generate any Ising H
        self.node_labels = {}                           # The labels for the nodes
        self.edge_labels = {}                           # The labels for the edges
        self.node_indices = {}                          # node -> index
        self.edge_indices = {}                          # edge -> index
        self.node_labels_style = "node_name"            # What to use as labels for the nodes
        self.E = []
        self.V = []
        self.tmax = 0.0
        
        # Set the graph family right away if specified
        if graph_family is not None:
            self.graph_family = graph_family
            
            # Generate the required terms
            for k, G in graph_family.items():
                self.setNthOrderGraph(G, k)
        
    def setNthOrderGraph(self, Gn, n):
        """ Sets a graph for the n-th order coupling terms of the Ising Hamiltonian. The graph for orders 1 and 2 should be the same, and any higher order graph can be the same or different. Also initialises the corresponding coefficients to the default 0 value.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be and integer greater than 0.")
        if not isinstance(Gn, nx.Graph):
            raise Exception("Specified graph should be a networkx.Graph instance.")
        
        nodes = list(Gn.nodes)
        edges = list(Gn.edges)
        
        # Order 1 and 2 are handled at the same time
        if n == 1 or n == 2:
            self.N_qubits = len(self.graph_family[1].nodes)
            self.graph_family[1] = Gn
            self.graph_family[2] = Gn
            self.node_indices = {nodes[i]:i for i in range(len(nodes))}
            #self.edge_indices[2] = {
            #    (
            #        self.node_indices[edges[i][0]],
            #        self.node_indices[edges[i][1]]
            #    ):i for i in range(len(edges))
            #}
            self.edge_indices[2] = {
                (
                    edges[i][0],
                    edges[i][1]
                ):i for i in range(len(edges))
            }
            
            # Generate the coefficient symbols
            terms1 = ["h"+"".join(x) for x in itt.product(['x','y','z'], repeat=(1))]
            terms2 = ["J"+"".join(x) for x in itt.product(['x','y','z'], repeat=(2))]
            
            # Init the coefs data structure for these
            coefs1 = {k:0.0 for k in terms1}
            coefs2 = {k:0.0 for k in terms2}
            
            # Assign for each node and edge
            self.coefs_of_order[1] = {(k,):coefs1.copy() for k,i in self.node_indices.items()}
            self.coefs_of_order[2] = {e:coefs2.copy() for e,i in self.edge_indices[2].items()}
            
        else:
            self.graph_family[n] = Gn
            
            # Get the cycles of order n
            cycles = self._find_cycles_of_length(n)
            index_cycles = [[self.node_indices[node] for node in c] for c in cycles]
            self.edge_indices[n] = {tuple(cycles[i]):i for i in range(len(cycles))}
            
            # Generate the coefficient symbols
            terms = ["J"+"".join(x) for x in itt.product(['x','y','z'], repeat=(n))]
            
            # Init the coefs data structure for these
            coefs = {k:0.0 for k in terms}
            
            # Init the coefs data structure
            self.coefs_of_order[n] = {e:coefs.copy() for e,i in self.edge_indices[n].items()}
    
    def getNthOrderGraph(self, n):
        """ Gets the n-th order graph.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        return self.graph_family[n]
    
    def setNthOrderCoefs(self, n, params):
        """ Sets the N local parameters of the Ising Hamiltonian.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        
        # Check the node and edges exist
        in_set = set(params.keys())
        total_set = set(self.coefs_of_order[n].keys())
        if not in_set <= total_set:
            raise Exception("Nodes or edges specified in 'params' do not exist.")
        
        # Check the parameters exist for each node and edge specified
        if n == 1:
            terms = ["h"+"".join(x) for x in itt.product(['x','y','z'], repeat=(n))]
        else:
            terms = ["J"+"".join(x) for x in itt.product(['x','y','z'], repeat=(n))]
        total_set = set(terms)
        for key, val in params.items():
            in_set = set(val.keys())
            if not in_set <= total_set:
                raise Exception("Parameter name for node or edge '%s' does not exist." % (repr(key)))
        
        # Update the parameters for each coordinate specified
        for coord in params.keys():
            self.coefs_of_order[n][coord].update(params[coord])
    
    def getNthOrderCoefs(self, n):
        """ Gets the N local parameters of the Ising Hamiltonian.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        return self.coefs_of_order[n]
    
    def getNthOrderTerm(self, term):
        """ Gets specific N local parameters of the Ising Hamiltonian.
        """
        return {k:v[term] for k,v in self.getNthOrderCoefs(len(term[1:])).items()}
    
    def setCoef(self, coord, term, value, schedule):
        """ Sets a given term in the full Hamiltonian
        """
        if type(coord) in [int, np.int64]:
            self.coefs_of_order[1][(coord,)][term] = (value, schedule)
            #self.coefs_of_order[1][(coord,)][term] = value
        elif type(coord) in [list, tuple]:
            self.coefs_of_order[len(coord)][tuple(coord)][term] = (value, schedule)
            #self.coefs_of_order[len(coord)][tuple(coord)][term] = value
    
    def getCoef(self, coord, term, s, params={}):
        """ Gets a given term in the full Hamiltonian
        """
        if type(coord) in [int, np.int64]:
            val = self.coefs_of_order[1][(coord,)][term][0]
            sch = self.coefs_of_order[1][(coord,)][term][1]
            return val*sch(s,params)
        elif type(coord) in [list, tuple]:
            val = self.coefs_of_order[len(coord)][tuple(coord)][term][0]
            sch = self.coefs_of_order[len(coord)][tuple(coord)][term][1]
            return val*sch(s,params)
    
    def getCoefValue(self, coord, term):
        """ Gets a given term in the full Hamiltonian
        """
        if type(coord) in [int, np.int64]:
            return self.coefs_of_order[1][(coord,)][term][0]
        elif type(coord) in [list, tuple]:
            return self.coefs_of_order[len(coord)][tuple(coord)][term][0]
    
    def getCoefSchedule(self, coord, term):
        """ Gets a given term in the full Hamiltonian
        """
        if type(coord) in [int, np.int64]:
            return self.coefs_of_order[1][(coord,)][term][1]
        elif type(coord) in [list, tuple]:
            return self.coefs_of_order[len(coord)][tuple(coord)][term][1]
    
    def getOperator(self, coord, term):
        """ Gets the operator associated with the specified coordinate and term.
        """
        kron_pos = [self.__paulis["i"]]*self.N_qubits
        if type(coord) in [int, np.int64]:
            kron_pos[self.node_indices[coord]] = self.__paulis[term[1]]
        elif type(coord) in [list, tuple]:
            for i, p in enumerate(coord):
                kron_pos[self.node_indices[p]] = self.__paulis[term[i+1]]
        return qt.tensor(*kron_pos)
    
    def getBitString(self, s, params={}, op='hz', state=0):
        """ Generates a bit string that indicates the direction of spins for a given state. '0' is down and '1' is up.
        """
        
        E,V = self.getHamiltonian(s, params).eigenstates()
        
        s = ["0"]*self.N_qubits
        for i in range(self.N_qubits):
            s[i] = str(int((1+(self.getOperator(i, op).overlap(V[state])).real)/2))
        return "".join(s)
    
    def getMagnetizationOp(self, op='hz'):
        """ Build the magnetization operator.
        """
        
        nodes = self.getNodes()
        sz0 = self.getOperator(nodes[0], op)
        I = qt.qeye(2**len(nodes))
        I.dims = sz0.dims
        HW = I - I
        for n in nodes:
            HW += self.getOperator(n, op)
        return HW/len(nodes)
    
    def getHammingWeightOp(self):
        """ Build the Hamming Weight operator associated with the problem Hamiltonian.
        """
        nodes = self.getNodes()
        sz0 = self.getOperator(nodes[0], 'hz')
        I = qt.qeye(2**self.N_qubits)
        I.dims = sz0.dims
        HW = I - I
        for n in nodes:
            HW += 0.5*(I - self.getOperator(n, 'hz'))
        return HW
    
    def getHamiltonian(self, s, params={}):
        """ Generates the final Hamiltonian from the supplied graphs.
        """
        
        # Handle units here? (use pycqed.Units)
        factor = 2*np.pi
        
        Hti = 0.0
        kron_pos = [self.__paulis["i"]]*self.N_qubits
        for order, coefs in self.coefs_of_order.items():
            for pos, coef in coefs.items():
                for sym, val in coef.items():
                    if val == 0.0:
                        continue
                    v, sched = val
                    
                    # Setup the tensor product
                    kron_pos = [self.__paulis["i"]]*self.N_qubits
                    for i,p in enumerate(pos):
                        pi = self.node_indices[p]
                        kron_pos[pi] = self.__paulis[sym[i+1]]
                    
                    # Compute value
                    Hti += v*sched(s, params)*qt.tensor(*kron_pos)
        return factor*Hti
    
    def _sched_decorator(self, func):
        def wrapper(t, args={}):
            tan = args['tan']
            return func(t/tan, args)
        return wrapper
    
    def getQobjEvo(self, params={}):
        # Handle units here? (use pycqed.Units)
        factor = 2*np.pi
        
        evo_list = []
        kron_pos = [self.__paulis["i"]]*self.N_qubits
        for order, coefs in self.coefs_of_order.items():
            for pos, coef in coefs.items():
                for sym, val in coef.items():
                    if val == 0.0:
                        continue
                    v, sched = val
                    
                    # Setup the tensor product
                    kron_pos = [self.__paulis["i"]]*self.N_qubits
                    for i,p in enumerate(pos):
                        pi = self.node_indices[p]
                        kron_pos[pi] = self.__paulis[sym[i+1]]
                    
                    # Wrap the schedule to now depend on time
                    sched_wrap = self._sched_decorator(sched)
                    
                    # Compute value
                    evo_list.append([factor*v*qt.tensor(*kron_pos), sched_wrap])
        return qt.QobjEvo(evo_list, args=params)
    
    def drawNthOrderGraph(self, n):
        """ Draws the specified graph.
        """
        if n <= 0:
            raise Exception("Specified order 'n' should be an integer greater than 0.")
        if n not in self.graph_family.keys():
            raise Exception("Specified order 'n=%i' does not have an associated graph." % n)
        
        # Make our own figure
        fig, ax = plt.subplots(1,1,constrained_layout=True,figsize=(7,7))
        
        # Get layout first as it is randomly generated
        #L = nx.spring_layout(self.graph_family[n])
        #L = nx.planar_layout(self.graph_family[n])
        #L = nx.kamada_kawai_layout(self.graph_family[n])
        L = nx.shell_layout(self.graph_family[n])
        
        self._update_node_labels()
        nx.draw(
            self.graph_family[n],
            pos=L,
            with_labels=True,
            node_color=self.__node_color,
            node_size=self.__node_size,
            width=self.__edge_size,
            labels=self.node_labels,
            font_weight=self.__label_font_weight,
            font_size=self.__label_font_size,
            ax=ax
        )
        
        self._update_edge_labels()
        nx.draw_networkx_edge_labels(
            self.graph_family[n],
            L,
            edge_labels=self.edge_labels,
            font_weight=self.__label_font_weight,
            font_size=self.__label_font_size,
            ax=ax
        )
    
    def getNodes(self):
        """ Gets the node names of the graph.
        """
        return list(self.graph_family[1].nodes)
    
    def getEdges(self, n=1):
        """ Gets the edge names of the graph.
        """
        if n == 1 or n == 2:
            return list(self.graph_family[1].edges)
        else:
            return list(self.coefs_of_order[n].keys())
    
    def getEigenEnergies(self, args={}):
        """ Get the eigenenergies of the Hamiltonian
        """
        if len(self.E) == 0:
            self._diagonalise_hamiltonian(args=args)
        return self.E
    
    def getEigenVectors(self, args={}):
        """ Get the eigenvectors of the Hamiltonian
        """
        if len(self.V) == 0:
            self._diagonalise_hamiltonian(args=args)
        return self.V
    
    ###################################################################################################################
    #       Internal Functions
    ###################################################################################################################
    
    def _diagonalise_hamiltonian(self, args={}):
        H = self.getHamiltonian()
        if type(H) is qt.Qobj:
            Hq = H
        else:
            Hq = qt.QobjEvo(H, args=args)(self.tmax)
        self.E, self.V = util.diagDenseH(Hq, eigvalues=Hq.shape[0], get_vectors=True)
    
    def _update_node_labels(self):
        
        # Get nodes
        nodes = self.getNodes()
        
        # Get parameters
        if self.node_labels_style == "node_name":
            for node in nodes:
                self.node_labels[node] = str(node)
        elif self.node_labels_style == "paulis":
            for node in nodes:
                s = ""
                for k,v in self.coefs_of_order[1][(node,)].items():
                    s += "%s=%.3f\n" % (k,v)
                self.node_labels[node] = s.strip('\n')
    
    def _update_edge_labels(self):
        
        # Get edges
        edges = self.getEdges()
        
        # Get parameters
        if self.node_labels_style == "node_name":
            for edge in edges:
                self.edge_labels[edge] = "(%i, %i)" % (edge[0],edge[1])
        elif self.node_labels_style == "paulis":
            pass
    
    def _find_cycles_of_length(self, n, source=None, cycle_length_limit=None):
        """Adapted from https://gist.github.com/joe-jordan/6548029
        
        To make this more efficient:
         - Need a way to stop looking for longer cycles than set by n
         - More effectively filter out the cycles of length n
         - Go from nodes to their indices
        
        """
        G = self.graph_family[n]
        if source is None:
            # produce edges for all components
            nodes=[list(i)[0] for i in nx.connected_components(G)]
        else:
            # produce edges for components with source
            nodes=[source]
        
        # extra variables for cycle detection:
        cycle_stack = []
        output_cycles = set()
        
        def get_hashable_cycle(cycle):
            """cycle as a tuple in a deterministic order."""
            m = min(cycle)
            mi = cycle.index(m)
            mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
            if cycle[mi-1] > cycle[mi_plus_1]:
                result = cycle[mi:] + cycle[:mi]
            else:
                result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
            return tuple(result)
        
        for start in nodes:
            if start in cycle_stack:
                continue
            cycle_stack.append(start)
            
            stack = [(start,iter(G[start]))]
            while stack:
                parent,children = stack[-1]
                try:
                    child = next(children)
                    
                    if child not in cycle_stack:
                        cycle_stack.append(child)
                        stack.append((child,iter(G[child])))
                    else:
                        i = cycle_stack.index(child)
                        if i < len(cycle_stack) - 2:
                          output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                    
                except StopIteration:
                    stack.pop()
                    cycle_stack.pop()
        
        unique_cycles = list(np.unique([sorted(list(i)) for i in output_cycles]))
        if type(unique_cycles[0]) in [int, np.int64]:
            return [unique_cycles]
        return [x for x in unique_cycles if len(x) == n]