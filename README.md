## Double-bracket quantum algorithms for quantum imaginary-time evolution

https://arxiv.org/abs/2412.04554

## About

Efficiently preparing approximate ground-states of large, strongly correlated systems on quantum hardware is challenging and yet nature is innately adept at this. This has motivated the study of thermodynamically inspired approaches to ground-state preparation that aim to replicate cooling processes via imaginary-time evolution. However, synthesizing quantum circuits that efficiently implement imaginary-time evolution is itself difficult, with prior proposals generally adopting heuristic variational approaches or using deep block encodings. Here, we use the insight that quantum imaginary-time evolution is a solution of Brockett's double-bracket flow and synthesize circuits that implement double-bracket flows coherently on the quantum computer. 
We prove that our Double-Bracket Quantum Imaginary-Time Evolution (DB-QITE) algorithm inherits the cooling guarantees of imaginary-time evolution. Concretely, each step is guaranteed to i) decrease the energy of an initial approximate ground-state by an amount proportion to the energy fluctuations of the initial state and ii) increase the fidelity with the ground-state. 
We provide gate counts for DB-QITE through numerical simulations in Qrisp which demonstrate scenarios where DB-QITE outperforms quantum phase estimation.
Thus DB-QITE provides a means to systematically improve the approximation of a ground-state using shallow circuits. 

## Authors 

Marek Gluza, Jeongrak Son, Bi Hong Tiang, René Zander, Raphael Seidel, Yudai Suzuki, Zoë Holmes, Nelly H. Y. Ng 
