# Mixed-Logic Line Decoders: 32nm High-Performance Implementation

This repository presents the **design, simulation, and performance analysis of low-power high-performance 2-4 and 4-16 line decoders** using a **mixed-logic design methodology**.

The project was developed for the **Mid-Semester Project Evaluation (AY 2025-2026)** at **National Institute of Technology Agartala**.

The work focuses on improving decoder efficiency by combining multiple logic styles instead of relying solely on conventional static CMOS implementations.

---

# Project Overview

Traditional CMOS decoders offer strong logic levels but require a relatively **high transistor count and increased switching activity**, which increases power consumption and layout area.

To address this, the proposed architecture integrates three logic styles:

- **Transmission Gate Logic (TGL)**
- **Pass-Transistor Dual-Value Logic (DVL)**
- **Static CMOS**

This **mixed-logic approach** reduces transistor usage while maintaining strong output levels where required. The result is a decoder architecture optimized for:

- Reduced silicon area  
- Lower power consumption  
- Improved energy efficiency  
- High-speed operation suitable for memory peripheral circuits

---

# System Architecture

## 2-4 Line Decoder (2-4HP Model)

The **2-4 High-Performance (2-4HP)** decoder forms the fundamental building block of the system.

### Key Characteristics

- **Total Transistors:** 15  
  - 9 nMOS  
  - 6 pMOS  

### Design Strategy

**Input Stage**

A single **static CMOS inverter** generates the complemented signal for input **A**, minimizing transistor overhead.

**Minterm D0**

Implemented using a **static CMOS NOR gate**.  
This stage ensures:

- Full voltage swing  
- Strong restoring capability  
- Reliable logic levels

**Minterms D1 – D3**

These outputs are implemented using **3-transistor mixed-logic AND structures** based on:

- Transmission Gate Logic  
- Dual-Value Pass-Transistor Logic

This allows significant transistor reduction while maintaining correct logical behavior.

**Logic Optimization**

Because of the asymmetric nature of the pass-transistor structures, the **B-input inverter is eliminated**, which reduces:

- Switching activity  
- Dynamic power consumption  
- Overall area

---

## 4-16 Line Decoder (4-16HP Model)

The larger decoder is implemented using a **predecoding architecture**, which improves scalability and performance.

### Structure

**Predecoder Stage**

Two **2-4HP mixed-logic decoders** generate intermediate signals.

**Post-Decoder Stage**

A **16-unit array of 2-input CMOS NOR gates** generates the final outputs.

### Transistor Count

- Conventional CMOS design: **104 transistors**
- Proposed mixed-logic design: **94 transistors**

This results in approximately **9.6% reduction in transistor count**, translating directly into area savings.

---

# Simulation Methodology

All circuits were validated using **SPICE-level simulations**.

### Simulation Environment

- **Simulator:** LTspice  
- **Technology Model:** 32nm Predictive Technology Model (PTM-LP)  
- **Supply Voltage:** 1.0 V  
- **Operating Frequency:** 1 GHz  

### Loading Conditions

Each output node includes a load capacitance of: 0.2 fF

### Verification Procedure

**2-4 Decoder**

A **64 ns transient simulation** verifies all input combinations.

**4-16 Decoder**

A **256 ns transient simulation** evaluates the entire switching space of the decoder.

These simulations capture:

- Dynamic switching behavior  
- Worst-case propagation delays  
- Average power consumption

---

# Experimental Results

The proposed mixed-logic design was compared against a conventional CMOS implementation.

| Parameter | Conventional CMOS (104T) | Proposed 4-16HP (94T) |
|---|---|---|
| **Transistor Count** | 104 | **94** |
| **Average Power** | 2.572 µW | **1.945 µW** |
| **Max Propagation Delay** | 88 ps | **191.5 ps** |
| **Power-Delay Product** | 226.33 aJ | 372.58 aJ |
| **Logic Swing** | Full-Swing | Full-Swing |
| **Architecture** | Standard CMOS | Mixed-Logic (TGL / DVL / CMOS) |

### Key Observations

- **9.6% reduction in transistor count**
- **24.36% reduction in average power consumption**
- Full-swing logic levels maintained
- Slight increase in delay due to pass-transistor structures

Despite the increased delay, the design achieves **significant power savings and area reduction**, making it attractive for **low-power high-density digital circuits**.

---

# Conclusion

The mixed-logic design methodology successfully combines the strengths of multiple logic families to create a **compact and energy-efficient decoder architecture**.

Key achievements include:

- Reduced transistor count
- Lower power consumption
- Area optimization
- Reliable full-swing outputs

The **4-16HP mixed-logic decoder** demonstrates that hybrid logic styles can outperform conventional CMOS in **power-constrained VLSI systems**, making it well suited for applications such as:

- Memory address decoding
- Embedded processors
- High-density digital systems

---

# Reference

D. Balobas and N. Konofaos  
**“Design of Low Power, High Performance 2-4 and 4-16 Mixed-Logic Line Decoders”**  
IEEE Transactions on Circuits and Systems II: Express Briefs, 2016.
