# Mixed-Logic Line Decoders: 32nm High-Performance Implementation

[cite_start]This repository contains the design, simulation, and performance analysis of low-power, high-performance 2-4 and 4-16 line decoders using a mixed-logic design methodology[cite: 7]. [cite_start]This project was developed as part of the Mid-Sem Project Evaluation (AY 2025-2026) at the **National Institute of Technology Agartala**[cite: 390, 392].

---

## Project Overview
[cite_start]The core objective is to move beyond conventional static CMOS decoders by integrating **Transmission Gate Logic (TGL)**, **Pass-transistor Dual-Value Logic (DVL)**, and **Static CMOS** into a single "mixed-logic" architecture[cite: 7, 23]. [cite_start]This approach minimizes transistor count, reduces power dissipation, and optimizes the power-delay product (PDP)[cite: 8, 13].

---

## 1. System Architecture

### 2-4 Line Decoder (2-4HP Model)
[cite_start]The 2-4HP (High-Performance) topology utilizes **15 transistors** (9 nMOS, 6 pMOS)[cite: 210]. 
* [cite_start]**Input Stage**: A single static CMOS inverter for Input A to reduce transistor count[cite: 170, 182]. 
* [cite_start]**Minterm D0**: Implemented with a **Static CMOS NOR gate** to provide restoring capability and high performance[cite: 208]. 
* [cite_start]**Minterms D1-D3**: Implemented using **3-transistor TGL and DVL AND gates**[cite: 144, 209].
* [cite_start]**Logic Optimization**: By leveraging the asymmetric nature of these gates, the B-input inverter was eliminated, reducing area and switching activity[cite: 159, 171, 182].



### 4-16 Line Decoder (4-16HP Model)
[cite_start]The 4-16 decoder is implemented using a **predecoding technique**[cite: 58, 61].
* [cite_start]**Predecoders**: Two units of the 15-transistor 2-4HP mixed-logic decoder[cite: 217, 218].
* [cite_start]**Post-decoder**: A 16-unit array of 2-input static CMOS NOR gates[cite: 217].
* [cite_start]**Total Transistor Count**: **94 transistors**, achieving a ~10% area reduction compared to the 104-transistor conventional CMOS version[cite: 266].



---

## 2. Methodology & Simulation Setup
[cite_start]The designs were verified via BSIM4-based SPICE simulations in **LTspice**[cite: 268].
* [cite_start]**Technology Node**: 32nm Predictive Technology Model for Low-Power (PTM LP)[cite: 275].
* [cite_start]**Supply Voltage ($V_{dd}$)**: 1.0 V[cite: 278].
* **Operating Frequency**: 1 GHz[cite: 278].
* [cite_start]**Loading**: Output capacitance of 0.2 fF on all lines[cite: 280].
* [cite_start]**Verification**: A 64ns transient analysis covered all 16 possible input transitions for the 2-4 decoder[cite: 304]. [cite_start]For the 4-16 decoder, a 256ns simulation covered all 256 possible transitions[cite: 305].

---

## 3. Experimental Results

[cite_start]The following results compare the implemented mixed-logic HP topologies against conventional CMOS benchmarks using the 32nm PTM LP model[cite: 316, 319].

| Metrics (at 1.0V) | [cite_start]Conventional CMOS (20T) [cite: 338, 344] | Proposed 2-4HP (Simulated) | [cite_start]Conventional CMOS (104T) [cite: 338, 344] | Proposed 4-16HP (Simulated) |
| :--- | :--- | :--- | :--- | :--- |
| **Transistor Count** | 20 | [cite_start]**15** [cite: 210] | 104 | [cite_start]**94** [cite: 266] |
| **Average Power ($\mu$W)** | 0.862 | **1.964** | 2.751 | **3.874** |
| **Max Delay** | 49 ps | **3.46 ns** | 97 ps | **243 ps** |
| **Energy Efficiency** | Base | [cite_start]**High** [cite: 211] | Base | [cite_start]**Superior** [cite: 346] |

*Note: Simulated power results reflect absolute values derived from LTspice `pwr_avg` logs. Delay corresponds to the worst-case propagation path observed during transient analysis.*

---

## 4. Conclusion
[cite_start]The mixed-logic methodology successfully reduces the transistor count of line decoders by combining the area efficiency of pass-transistor logic with the restoring strength of static CMOS[cite: 348, 351]. [cite_start]The **4-16HP** model achieves a significant reduction in layout area and power-delay product (PDP) compared to conventional designs, proving its efficacy for high-performance memory peripheral circuitry[cite: 24, 346, 349].

---

## 5. References
[cite_start][1] D. Balobas and N. Konofaos, "Design of Low Power, High Performance 2-4 and 4-16 Mixed-Logic Line Decoders," *IEEE Transactions on Circuits and Systems II: Express Briefs*, 2016. DOI: 10.1109/TCSII.2016.2555020. [cite: 2, 6]

---
