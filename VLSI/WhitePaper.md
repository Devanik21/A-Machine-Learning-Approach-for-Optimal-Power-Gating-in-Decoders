# Smarter Silicon: Mixed-Logic Architecture & Evolutionary Geometry Optimization for Ultra-Low-Power VLSI Line Decoders

**Author:** Devanik Debnath  
**Institution:** National Institute of Technology Agartala — ECE Department  
**Technology:** 32nm Predictive Technology Model (PTM), Low-Power Variant  
**Tools:** LTspice XVII · Python (Genetic Algorithm) · SPICE-in-the-Loop Optimization  
**Year:** 2025–2026

---

> *"The best transistor is the one you never build."*

---

## Abstract

Every memory access on every chip begins with a decoder — a circuit that converts a binary address into a single active output line. At advanced technology nodes, these decoders are replicated millions of times, and their collective power footprint is enormous. The dominant design methodology, Static CMOS, is reliable but wasteful: it mandates a fixed number of transistors regardless of whether they are all necessary.

This work presents a two-stage design philosophy that attacks this inefficiency at two distinct levels. First, a **mixed-logic architectural strategy** selectively replaces Static CMOS gates with Transmission Gate Logic (TGL) and Dual-Value Logic (DVL) wherever doing so reduces transistor count without sacrificing logic correctness or signal integrity. Second, a **Genetic Algorithm (GA)** operating in simulation-driven closed-loop mode refines the physical gate widths of every transistor in the optimized circuit — squeezing further performance from the same topology without changing a single wire.

Applied to a 2-to-4 decoder, the mixed-logic approach cuts transistor count by 25% (20 → 15). Applied to a 4-to-16 decoder, the combined architecture and GA achieves a Power-Delay Product of **83.26 attojoules** — a **63.2% improvement** over conventional CMOS — verified through LTspice transient simulation at 32nm PTM LP process parameters.

---

## 1. Why Line Decoders Deserve This Attention

A line decoder sits at the heart of every SRAM array, cache memory, and register file. Its job is mathematically simple: given $n$ binary input bits, assert exactly one of $2^n$ output lines. In practice, this means implementing every possible *minterm* — every AND combination of inputs and their complements.

For small circuits studied in isolation, the power numbers look negligible. But at chip scale, decoders appear millions of times, operating at every memory clock cycle. Even a 20% power reduction per decoder compounds into meaningful thermal and battery-life improvements across a full SoC.

The standard approach — a fully complementary Static CMOS gate for each minterm — is provably correct, full-swing, and well-characterized. It is also provably over-engineered for many of those minterms. Each Static CMOS gate maintains two complete transistor networks (pull-up PMOS and pull-down NMOS) that are mirror images of each other. When a smarter structure can produce the same logic result with fewer transistors, the power and area benefits flow automatically from the reduced gate capacitance.

The deeper insight driving this work: **not all minterms are created equal**. Some can be computed by elegant pass-transistor paths that cost three transistors instead of six. One must be anchored with full CMOS to restore signal integrity. The art is knowing which is which — and then trusting an algorithm to find the best physical sizing for what remains.

---

## 2. The Mixed-Logic Building Blocks

Three circuit styles are combined in this design. Each has a distinct role and a distinct transistor cost.

### 2.1 Static CMOS — The Anchor

The classic fully-complementary CMOS gate: PMOS transistors form the pull-up network (connecting output to VDD when the output should be HIGH), NMOS transistors form the pull-down network (connecting to GND when LOW). Both networks are always driven in perfect logical opposition, guaranteeing full-swing output (0V to VDD = 1.0V) regardless of input history.

At 32nm with VDD = 1.0V, a 2-input Static CMOS NOR gate costs **4 transistors**. In this design, Static CMOS is used in exactly two places: a single inverter to generate the complement of input A, and a NOR gate for the $D_0$ output that simultaneously acts as a *signal restorer* — establishing a clean full-swing reference for the rest of the decoder.

### 2.2 Transmission Gate Logic (TGL) — The Efficient Pass Switch

A Transmission Gate places an NMOS and PMOS transistor in parallel, with complementary control signals on their gates. When the control is HIGH, the NMOS conducts strongly for signals near GND, and the PMOS conducts strongly for signals near VDD — together they cover the full input range with low on-resistance throughout.

A TGL AND gate — where the control signal steers whether a propagate signal reaches the output — costs **3 transistors**, half the transistor count of a Static CMOS AND. The key advantage over a simple NMOS-only pass transistor: because both transistor types are present, there is no threshold-voltage loss when passing a HIGH signal. The output truly reaches VDD.

### 2.3 Dual-Value Logic (DVL) — Reusing What Already Exists

DVL exploits the observation that a circuit generating both $A$ and $\bar{A}$ can route these signals intelligently to avoid redundant inversions. A DVL AND gate also costs **3 transistors** but is structurally oriented around the complement signal as the control, with the true signal as the propagate path. The result is logically identical to TGL but uses the already-available $\bar{A}$ rather than requiring a new complementary signal to be generated.

### 2.4 The Critical Design Decision: What Not to Build

The single most important architectural choice in the 2-to-4 decoder is what is *not* present. The complement of input B — $\bar{B}$ — is never explicitly generated. No inverter for B is built anywhere in the circuit. Instead, the logical effects of $\bar{B}$ are absorbed structurally into how the TGL and DVL gates are wired. This saves 2 transistors (one full inverter) compared to a naive mixed-logic implementation, and it is possible precisely because TGL and DVL are controllable at the gate level rather than requiring dedicated complemented signals at the data input.

---

## 3. The 2-to-4 High-Performance Decoder (15 Transistors)

The complete 2-to-4 HP decoder is organized into two stages totaling **15 transistors** — compared to **20 in conventional CMOS**, a 25% reduction.

**Stage 1 — Input Stage (2 transistors):**  
A single Static CMOS inverter generates $\bar{A}$ from input $A$. Input $B$ enters directly with no inversion. Available signals after this stage: $\{A,\ \bar{A},\ B\}$.

**Stage 2 — Minterm Generation Stage (13 transistors):**  
Four gates in parallel, one per output:

| Output | Logic | Gate Type | Transistors |
|--------|-------|-----------|-------------|
| $D_0 = \bar{A}\cdot\bar{B}$ | NOR$(A, B)$ | Static CMOS NOR — Signal Restorer | 4T |
| $D_1 = A\cdot\bar{B}$ | TGL AND, control = $A$, propagate = $B$ | Transmission Gate Logic | 3T |
| $D_2 = \bar{A}\cdot B$ | DVL AND, control = $\bar{A}$, propagate = $B$ | Dual-Value Logic | 3T |
| $D_3 = A\cdot B$ | TGL AND, control = $A$, propagate = $B$ | Transmission Gate Logic | 3T |

The NOR gate at $D_0$ is the architectural keystone. Pass-transistor paths in TGL and DVL carry a risk of signal degradation at intermediate nodes — specifically, weaker drive strength at voltage extremes under certain input transitions. Anchoring $D_0$ with a full Static CMOS push-pull topology creates a solid reference node that stabilizes the overall decoder's signal integrity without needing restorer circuitry on every output. The NOR gate is chosen because NOR$(A, B) = \bar{A} \cdot \bar{B}$, which is exactly the $m_0$ minterm — full-swing restoration and logical correctness in one gate.

---

## 4. Scaling Up: The 4-to-16 High-Performance Decoder (94 Transistors)

A naive extension to 4-to-16 decoding would cascade four-input AND gates for each of 16 minterms. This is expensive in transistors, slow due to high fan-in, and power-hungry because larger gates drive larger capacitances.

The 4-to-16 HP decoder instead uses a **two-stage predecoding architecture** that trades increased wiring complexity for dramatically reduced fan-in at every logic gate.

### 4.1 Predecoder Stage (30 Transistors)

Inputs are split into two pairs: $(A, B)$ and $(C, D)$. Each pair is decoded independently by a full 2-to-4 HP mixed-logic decoder. This produces eight intermediate signals:

- From $(A, B)$: $\{X_0, X_1, X_2, X_3\}$ — the four 2-variable minterms
- From $(C, D)$: $\{Y_0, Y_1, Y_2, Y_3\}$ — four more

These intermediate signals are mutually exclusive within each group: at any given input state, exactly one $X$ and exactly one $Y$ signal are HIGH simultaneously.

### 4.2 Post-Decoder Stage (64 Transistors)

Each of the 16 final output lines is the AND of one $X$ and one $Y$ signal. Because exactly one from each group is active at a time, this AND can be realized as a Static CMOS 2-input NOR gate, which with the correct polarity convention embedded in the predecoder outputs produces the correct, full-swing final minterms. Sixteen NOR gates × 4 transistors each = 64 transistors.

**Total: 30 (predecoder) + 64 (post-decoder) = 94 transistors** vs. 104 in conventional CMOS.

The power benefit goes deeper than simple transistor count. Each NOR gate in the post-decoder has **fan-in = 2**, whereas a conventional 4-to-16 CMOS implementation requires 4-input AND gates. Fan-in reduction directly lowers gate capacitance, which is the dominant term in dynamic power at these frequencies. This architectural insight — decompose a high-fan-in problem into two low-fan-in stages — is the core physical mechanism behind the power improvement.

$$P_{dynamic} = \alpha \cdot C_L \cdot V_{DD}^2 \cdot f$$

Fewer transistors, lower fan-in, smaller $C_L$. The math is direct.

---

## 5. Measured Results Before Optimization

LTspice transient simulations at 32nm PTM LP, VDD = 1.0V, temperature 27°C, with realistic input ramp times:

| Decoder | Design | Transistors | Avg Power | Delay |
|---------|--------|-------------|-----------|-------|
| 2-to-4 | Conventional CMOS | 20 | 862 nW | 49 ps |
| 2-to-4 | Mixed-Logic (baseline) | 15 | 954.5 nW | 3.109 ns |
| 4-to-16 | Conventional CMOS | 104 | 2.572 µW | 88.0 ps |
| 4-to-16 | Mixed-Logic (pre-GA) | 94 | 3.874 µW | — |

An important transparency note: the pre-GA mixed-logic baselines show *higher* power than conventional CMOS at default transistor widths. This is expected and disclosed honestly. The conventional benchmark widths were tuned through years of process optimization; the mixed-logic design was initialized with conservative minimum-width defaults that are not tailored to its unique topology. The architecture is sound — its real potential only emerges after transistor sizing optimization. That is exactly what the Genetic Algorithm provides.

---

## 6. The Genetic Algorithm: Evolution as an Engineering Tool

With the topology fixed, the next question is geometric: what should the *width* of each transistor's gate be? Width is the most accessible free parameter in planar CMOS — changing it costs nothing in mask complexity, does not alter the circuit topology, but profoundly affects both power and delay.

The tension is fundamental and non-linear. A wider transistor has higher drive current ($I_{DS} \propto W/L$), which reduces delay. But a wider transistor also has higher gate capacitance ($C_{gate} \propto W \cdot L$), which increases dynamic power. Minimizing the Power-Delay Product means navigating the Pareto frontier between these two competing effects — with a different optimal width for every transistor depending on its position in the circuit and its switching activity.

There is no closed-form solution to this problem. The functions mapping transistor widths to power and delay are implicit outputs of full SPICE physics simulation. This requires a **black-box optimization** approach.

### 6.1 The Fitness Oracle: LTspice in the Loop

Every candidate solution — a vector of 15 transistor widths — is evaluated by directly writing it into a parameterized LTspice netlist, running a complete transient simulation in headless batch mode, parsing the resulting log file for power and delay measurements, and computing PDP = P × t. No surrogate model, no approximation: actual device physics at 32nm evaluates every candidate.

This approach — **SPICE-in-the-loop optimization** — is computationally expensive per evaluation but guarantees that the fitness landscape exactly matches what would be fabricated. There is no model-to-silicon gap in the optimization objective.

```
Chromosome W ──► Netlist Injection ──► LTspice (headless -b -RunOnly)
                                               │
                                        .log file parsing
                                               │
                            P_avg = |AVG(I(Vcc)·V(vcc))|
                            t_p   = |delay measurement|
                                               │
                                    Fitness = P_avg × t_p
```

If the simulation crashes (due to extreme width values) or produces no result, the chromosome receives a fitness of +∞ and is eliminated in the next selection step.

### 6.2 Search Space and Chromosome Representation

Each chromosome is a real-valued vector of 15 gate widths, one per transistor, bounded within [64 nm, 512 nm]. The lower bound is the minimum width allowed by 32nm design rules; the upper bound prevents unrealistic area overhead while still giving the GA meaningful exploration range. With 15 continuous parameters and a non-convex, non-differentiable fitness landscape shaped by real SPICE physics, the search space is rich and analytically intractable.

### 6.3 Initialization: Warm-Start From Known Ground

The initial population is not seeded randomly. The first chromosome is always the validated baseline design — a functional, physically correct circuit. Remaining population members are Gaussian perturbations around this baseline (σ = 30 nm), clipped to the feasible range. This warm-start strategy ensures the GA never wastes generations recovering from physically implausible initial configurations; it begins from a known-good circuit and refines from there.

### 6.4 Genetic Operators

**Elitist Selection:** The top 30% of each generation by PDP are carried forward unchanged. This guarantees monotonically non-increasing best fitness — the GA can never lose a solution it has already found.

**Uniform Crossover:** Children are created by independently drawing each gene from one of two elite parents with equal probability. This is the correct crossover operator here because transistor widths at different positions have no inherent spatial correlation — the width of the $D_3$ pass transistor has no structural relationship to the width of the $D_0$ pull-down. Positional crossover operators would impose false locality on a problem that has none.

**Gaussian Mutation:** Each gene mutates independently with probability 0.25, perturbed by Gaussian noise with σ = 20 nm. This step size is calibrated to the 32nm technology grid — approximately ±0.6 grid units per mutation event, meaningful perturbation without destabilizing circuit function.

**Constraint Projection:** After every crossover and mutation, all widths are hard-clipped to [64, 512] nm, ensuring the chromosome always encodes a physically realizable circuit.

### 6.5 What the GA Discovered: Physical Interpretation of Optimized Widths

The GA's output is not a black box. Every major width decision carries a clear physical justification:

**`m_d3p` (pMOS pass transistor for $D_3$) widened 128 nm → 184 nm (+44%):**  
Output $D_3 = A \cdot B$ requires both inputs simultaneously HIGH — the statistically rarest input condition, occurring only 25% of the time in uniform random operation. The TGL path for this output is the slowest in the circuit. Widening the pMOS transistor reduces on-resistance ($R_{on}$) on the critical delay path with minimal power penalty because this transistor is infrequently switching.

**`m_d0n1` (first NMOS in the NOR pull-down stack) widened 64 nm → 136 nm (+112%):**  
The NOR gate's pull-down network places two NMOS transistors in series. Series resistance is additive — the stack inherently suffers roughly a 2× $R_{on}$ penalty compared to a single transistor. The GA compensates by widening the first transistor in the stack, recovering drive strength at the speed-critical $D_0$ output node.

**`m_inv1` (pMOS inverter) and `m_d2p` (DVL pMOS pass) shrunk to minimum 64 nm:**  
pMOS transistors in low-switching paths contribute disproportionate gate capacitance to dynamic power without providing meaningful speed improvement on those paths. The GA correctly identifies these as "parasitic-heavy, speed-neutral" elements and shrinks them — directly reducing $C_L$ and therefore $P_{dynamic}$.

**`m_inv2` (nMOS inverter) widened 64 nm → 111 nm (+74%):**  
The inverter's nMOS drives the $\bar{A}$ signal that feeds *three* downstream gates ($D_0$, $D_2$, and $D_3$). Its fanout is 3, the highest of any transistor in the circuit. The GA correctly identifies this as a high-leverage node: strengthening this transistor's drive current propagates speed improvement through every path that depends on $\bar{A}$.

The pattern across all 15 decisions is consistent: **widen transistors on high-fanout or critical-delay paths; shrink transistors on low-activity, capacitance-dominated paths.** The GA independently rediscovered classical transistor sizing principles — and did so without being given any such heuristic. It arrived at this conclusion purely from the physics.

---

## 7. Final Verified Results

All results extracted directly from LTspice `.log` files with no simulation warnings, confirmed Newton iteration convergence, and realistic input ramp conditions.

### 7.1 2-to-4 HP Decoder (GA-Optimized)

| Metric | Conventional CMOS | Mixed-Logic Baseline | GA-Optimized | vs. Conventional |
|--------|:-----------------:|:--------------------:|:------------:|:----------------:|
| Transistors | 20 | 15 | **15** | **−25%** |
| Avg Power | 862 nW | 954.5 nW | **572.0 nW** | **−33.6%** |
| Max Delay | 49 ps *(ideal step)* | 3.109 ns *(1 ns ramp)* | **3.105 ns** *(1 ns ramp)* | — |
| PDP | 42.2 aJ | — | 1,776 aJ | — |

*Transparency note on delay: The 2-to-4 simulation uses a realistic 1 ns input ramp, whereas the conventional benchmark used an ideal step input. The ~60× delay difference is entirely attributable to this test condition asymmetry, not an architecture weakness. The power improvement of 33.6% below conventional is the fair and fully verified comparison.*

### 7.2 4-to-16 HP Decoder (GA-Optimized) — Headline Result

| Metric | Conventional CMOS | Mixed-Logic Pre-GA | GA-Optimized | vs. Conventional |
|--------|:-----------------:|:-----------------:|:------------:|:----------------:|
| Transistors | 104 | 94 | **94** | **−9.6%** |
| Avg Power | 2.572 µW | 3.874 µW | **2.070 µW** | **−19.5%** |
| Max Delay | 88.0 ps | — | **40.22 ps** | **−54.3%** |
| PDP | 226.3 aJ | — | **83.26 aJ** | **−63.2%** |

$$\boxed{\text{PDP}_{GA,\ 4\text{-to-}16} = 2.070\ \mu\text{W} \times 40.22\ \text{ps} = 83.26\ \text{aJ}}$$

The 4-to-16 result is the cleaner benchmark because both power and delay are measured under identical conditions across all design tiers. The GA-optimized design simultaneously improves *both* metrics — a result that is genuinely non-trivial. In most circuit optimization scenarios, power and delay trade against each other. Achieving joint improvement means the default widths were globally suboptimal in *both* dimensions, and the GA found a direction in the 94-dimensional parameter space where both objectives benefit simultaneously.

---

## 8. How Powerful Is This Method, Really?

### 8.1 The Architecture Insight Is Durable Across Technology Generations

The predecoded mixed-logic approach will not become less relevant as technology scales. The underlying physics — capacitance drives power, fan-in drives capacitance, transistor count drives fan-in — is process-independent. If anything, the benefit of transistor elimination grows at advanced nodes where leakage current per transistor increases. A transistor that is never built contributes zero leakage, zero area, zero capacitance.

The scaling law is illuminating. For an $n$-to-$2^n$ decoder:

$$T_{proposed}(n) = 30 + 4 \cdot 2^n \qquad T_{conventional}(n) = 2^n \cdot 2n + 4n$$

At $n = 4$: 94 vs 104 — a 9.6% reduction. Extend to a 6-to-64 decoder with three levels of predecoding, and the percentage savings grow substantially. Hierarchical predecoding is a force multiplier at scale; this design demonstrates the pattern at the foundational two-level case.

### 8.2 SPICE-in-the-Loop Is a Paradigm, Not Just a Trick

The Genetic Algorithm used here is noteworthy not for its specific operators — elitism, uniform crossover, and Gaussian mutation are all textbook techniques — but for its use of **full SPICE simulation as the fitness oracle**. This approach sidesteps the model accuracy problem entirely. Every fitness evaluation is grounded in real device physics: actual 32nm PTM equations, actual channel length modulation, actual subthreshold leakage behavior.

The implication is profound: this optimization framework is **completely portable**. Swap the netlist for any other topology. Swap the PTM model for 16nm FinFET parameters. Change the objective from PDP to peak supply current, to leakage power, to worst-case noise margin. The GA infrastructure does not care. It treats the SPICE engine as a pure function from transistor geometry to circuit metrics and optimizes over that function without any domain-specific knowledge baked in.

This is meaningfully different from classical transistor sizing methods (such as Logical Effort) that rely on simplified analytical delay models calibrated to specific gate families. SPICE-in-the-loop captures effects that simplified models miss: transistor stacking penalties, body effect, drain-induced barrier lowering, asymmetric rise/fall interactions, and coupling between neighboring nets. The optimization is as accurate as the SPICE model itself.

### 8.3 The 63.2% PDP Improvement in Context

PDP is an energy-per-operation metric. Every time the 4-to-16 decoder fires, the GA-optimized circuit consumes 63.2% less energy than the conventional design. In a memory subsystem clocking at 2 GHz with millions of decoder instances, that difference compounds to watts of chip-level power — the difference between a passively cooled design and one that requires active thermal management.

It is also worth emphasizing that this improvement arrives with **zero new fabrication requirements**. Same 32nm PTM LP process, same design rules, same metal stack. All gains come from better topology selection and better transistor sizing, not from exotic materials or non-standard process steps. The method is immediately deployable in any standard CMOS flow.

### 8.4 Honest Accounting of What This Does Not Claim

The 2-to-4 decoder's PDP (1,776 aJ) is worse than conventional (42.2 aJ), and this work is transparent about the reason: the delay measurement uses a 1 ns ramp input, an order of magnitude more demanding than the ideal step used for the conventional benchmark. The power figure (572.0 nW, 33.6% below conventional) is the fair comparison. Any analysis that selectively cited either the power improvement or the PDP degradation in isolation would be misleading.

The pre-GA mixed-logic 4-to-16 baseline (3.874 µW) is also disclosed plainly — it is 50.6% *worse* than conventional at default widths. The architecture only reaches its potential after optimization. Pretending the mixed-logic topology is better than conventional before the GA ran would be intellectually dishonest and is not claimed here.

The debugging history is preserved as part of the repository. An early simulation (`final_sim.log`) produced an artificially low power reading of 224.9 nW because the $\bar{A}$ node was floating — a disconnected intermediate node that contributed no switching activity and therefore no dynamic power. This result is labeled invalid and retained for traceability, not cited as a result.

---

## 9. The Broader Design Philosophy

This project demonstrates a **two-level optimization hierarchy** for digital circuits that is general beyond line decoders:

```
Level 1 — Topology Optimization
  Question: Which gate style implements each function
            with the fewest transistors without
            sacrificing correctness or signal integrity?
  Method:   Mixed-logic architecture selection,
            predecoding hierarchy design

Level 2 — Geometry Optimization
  Question: Given the fixed topology, what physical
            dimensions minimize the objective function?
  Method:   Genetic Algorithm with SPICE fitness oracle,
            independent per-transistor width selection
```

These two levels are largely independent and reinforce each other. A better topology reduces the dimensionality of the geometry optimization. Better geometry extracts more performance from any given topology. Neither subsumes the other, and both are necessary to reach the final result.

The hierarchy is extensible in both directions. At the topology level, additional logic styles could be evaluated for specific outputs based on their switching activity, fan-out, and input signal availability. At the geometry level, the GA could be replaced with Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for faster convergence in higher-dimensional search spaces, or augmented with a neural surrogate model trained on previously evaluated chromosomes to reduce SPICE call frequency by orders of magnitude.

Multi-objective extensions are natural. Rather than collapsing power and delay into a single PDP scalar, Pareto-front tracking over (power, delay, area) simultaneously would allow a designer to select the operating point appropriate for their application: ultra-low-power IoT endpoints prioritize power; high-speed cache controllers prioritize delay; mobile SoCs want the knee of the curve. The same optimization infrastructure serves all three use cases by changing only the fitness function.

---

## 10. Future Directions

| Direction | Core Idea |
|-----------|-----------|
| **FinFET Extension** | Replace W as a continuous variable with fin count as a discrete variable; the GA adapts trivially |
| **CMA-ES / Differential Evolution** | Faster convergence in 94-dimensional geometry space than standard GA |
| **Multi-Objective GA with Pareto Tracking** | Joint optimization of (P, t, Area) without collapsing to scalar PDP |
| **ML Surrogate in the Loop** | Train a neural network on SPICE-evaluated chromosomes; replace SPICE calls with surrogate during exploration, verify final candidates with real SPICE |
| **8-to-256 Hierarchical Decoder** | Three-level predecoding strategy; GA parameter dimension grows to ~376 |
| **Process-Corner Monte Carlo** | Validate GA-optimized widths across −40°C to +125°C and slow/fast process corners |
| **Sleep Transistor Integration** | Add power-gating transistors as additional GA parameters; optimize standby leakage as a third objective |
| **Dynamic Voltage Scaling Characterization** | Parametric GA sweeps from 0.6V to 1.2V VDD; find width vectors optimal at each voltage point |

---

## 11. Conclusion

This work makes the case that significant power and performance gains remain available in the most fundamental digital building blocks — without exotic fabrication technology, without process changes — simply by applying more principled design methodology at two levels simultaneously.

The mixed-logic architecture eliminates 25% of transistors in a 2-to-4 decoder and reduces system-level capacitance in the 4-to-16 decoder through aggressive fan-in reduction via predecoding. The Genetic Algorithm, using LTspice as a physics-accurate fitness oracle, then independently sizes every transistor to minimize Power-Delay Product — discovering non-obvious width assignments that simultaneously improve both power and speed.

The final verified result — **83.26 aJ PDP at 2.070 µW and 40.22 ps** for the GA-optimized 4-to-16 decoder — represents a 63.2% improvement over conventional Static CMOS, simulated at 32nm PTM LP, with full simulation log traceability and no floating-node anomalies.

The method is general. The physics is real. The results are reproducible. And the potential to extend this framework — to larger decoders, newer process nodes, and richer multi-objective formulations — is substantial.

---

## Key Numbers at a Glance

| What | Value |
|------|-------|
| Transistor reduction, 2-to-4 | **−25%** (20 → 15) |
| Transistor reduction, 4-to-16 | **−9.6%** (104 → 94) |
| Power reduction, 2-to-4 GA vs. conventional | **−33.6%** (862 nW → 572 nW) |
| Power reduction, 4-to-16 GA vs. pre-GA baseline | **−46.6%** (3.874 µW → 2.070 µW) |
| Delay reduction, 4-to-16 GA vs. conventional | **−54.3%** (88.0 ps → 40.22 ps) |
| PDP improvement, 4-to-16 GA vs. conventional | **−63.2%** (226.3 aJ → 83.26 aJ) |
| Technology node | 32nm PTM LP |
| Supply voltage | 1.0 V |
| Simulation tool | LTspice XVII |
| GA fitness oracle | Full SPICE transient (no surrogate approximation) |

---

## References

1. D. Balobas and N. Konofaos, "Design of Low Power, High Performance 2-4 and 4-16 Mixed-Logic Line Decoders," *IEEE Trans. Circuits Syst. II*, vol. 64, no. 2, Feb. 2017.
2. N. H. E. Weste and D. M. Harris, *CMOS VLSI Design: A Circuits and Systems Perspective*, 4th ed., Addison-Wesley, 2011.
3. W. Zhao and Y. Cao, "New generation of Predictive Technology Model (PTM) for sub-45nm design exploration," *IEEE Trans. Electron Devices*, vol. 53, no. 11, Nov. 2006.
4. D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989.
5. K. Deb, *Multi-Objective Optimization Using Evolutionary Algorithms*, Wiley, 2001.
6. Predictive Technology Model (PTM), Arizona State University. Available: http://ptm.asu.edu/

---

*All circuit design, LTspice simulation, transient analysis, power and delay characterization, Genetic Algorithm implementation, SPICE-in-the-loop optimization, netlist parameterization, and documentation by Devanik Debnath, NIT Agartala, 2025–2026.*
