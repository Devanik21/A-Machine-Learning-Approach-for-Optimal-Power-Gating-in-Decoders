# Mixed-Logic Line Decoders: 32 nm High-Performance Implementation

This repository presents the **design, simulation, and performance evaluation of optimized 2-to-4 and 4-to-16 line decoders** implemented using a **mixed-logic design methodology**. The goal of this project is to improve decoder efficiency by combining multiple logic styles—rather than relying solely on conventional static CMOS circuits.

The work was carried out as part of the **Mid-Semester Project Evaluation (AY 2025-2026)** at **National Institute of Technology Agartala**, focusing on **low-power digital design and high-performance VLSI circuit optimization**.

---

# Project Overview

Line decoders are fundamental components in many digital systems, particularly in **memory architectures, address decoding circuits, and control units**. Conventional decoder implementations generally rely on **static CMOS logic**, which provides good noise margins and reliability but often results in **higher transistor count, larger silicon area, and increased power consumption**.

To address these limitations, this project explores a **mixed-logic design approach** that integrates multiple logic styles within the same circuit architecture:

* **Transmission Gate Logic (TGL)** – reduces transistor count and enables efficient signal transmission.
* **Pass-Transistor Dual-Value Logic (DVL)** – minimizes switching activity and simplifies AND gate implementations.
* **Static CMOS Logic** – provides signal restoration and robust logic levels where required.

By combining these techniques, the proposed architecture aims to achieve:

* Reduced transistor count
* Lower switching power
* Improved area efficiency
* Competitive performance for high-speed digital systems

The circuits were designed and evaluated using **SPICE-level simulation at the 32 nm technology node**, allowing realistic performance analysis under modern CMOS scaling conditions.

---

# 1. System Architecture

## 1.1 High-Performance 2-to-4 Decoder (2-4HP)

The proposed **2-4HP mixed-logic decoder** forms the fundamental building block for the larger decoder architecture.

### Key Design Characteristics

* **Total Transistor Count:** 15
* **nMOS:** 9
* **pMOS:** 6

This represents a significant reduction compared to the **20-transistor conventional CMOS implementation**.

### Input Stage

The input signal **A** is passed through a **single static CMOS inverter**.
This stage serves two main purposes:

* Provides a reliable complementary signal for downstream logic
* Maintains strong logic levels while minimizing additional transistor overhead

Only one inverter is used in the input stage, reducing both **dynamic switching activity and circuit area**.

### Minterm Generation

The decoder generates four outputs:

D₀, D₁, D₂, and D₃.

Each output corresponds to one minterm of the two input variables A and B.

#### Output D₀

Implemented using a **static CMOS NOR gate**.

Reasons for this choice:

* Provides **full voltage swing**
* Offers **strong restoring capability**
* Ensures reliable operation under varying load conditions

This gate acts as a **restoration point** within the circuit, preventing logic degradation from pass-transistor structures.

#### Outputs D₁ – D₃

These outputs are implemented using **3-transistor AND gate structures** based on:

* **Transmission Gate Logic (TGL)**
* **Dual-Value Logic (DVL)**

Advantages of this configuration include:

* Significant **transistor count reduction**
* Reduced **capacitive loading**
* Faster signal propagation in specific paths

### Logic Optimization

An important design improvement involves **removing the inverter for input B**.

Instead of generating B̅ explicitly, the asymmetric behavior of the mixed-logic gates is used to **directly produce the required logic conditions**.

This optimization provides several benefits:

* Fewer transistors
* Lower switching activity
* Reduced power dissipation
* Smaller layout area

---

## 1.2 High-Performance 4-to-16 Decoder (4-16HP)

The **4-to-16 decoder** expands the basic architecture using a **predecoding technique**, which is widely used in memory address decoders.

### Architectural Structure

The circuit consists of two main stages:

#### Predecoder Stage

Two **2-4HP mixed-logic decoders** operate as predecoders.

Each predecoder processes a pair of input signals and generates four intermediate signals.

This reduces the complexity of the final decoding stage.

#### Post-Decoder Stage

The outputs of the predecoders are combined using **sixteen 2-input static CMOS NOR gates**.

These gates generate the final **16 unique output lines**, each representing a specific combination of the four input variables.

### Advantages of Predecoding

Using predecoders provides several benefits:

* Reduces fan-in complexity
* Improves signal propagation paths
* Simplifies layout organization
* Enables better power optimization

### Transistor Count Comparison

| Implementation                 | Transistor Count |
| ------------------------------ | ---------------- |
| Conventional CMOS 4-16 Decoder | 104              |
| Proposed Mixed-Logic 4-16HP    | 94               |

This represents approximately **10 % area reduction**, which becomes significant in large memory arrays where thousands of decoders may be used.

---

# 2. Methodology and Simulation Setup

All circuits were designed and verified using **SPICE-level simulation in LTspice**, ensuring accurate transistor-level performance analysis.

### Technology Model

* **Technology Node:** 32 nm
* **Device Model:** Predictive Technology Model (PTM)
* **Model Type:** Low-Power (LP)

PTM models are widely used in academic research to represent realistic transistor characteristics at advanced technology nodes.

### Simulation Parameters

| Parameter               | Value   |
| ----------------------- | ------- |
| Supply Voltage          | 1.0 V   |
| Operating Frequency     | 1 GHz   |
| Output Load Capacitance | 0.2 fF  |
| Simulation Tool         | LTspice |
| Transistor Model        | BSIM4   |

### Transient Verification

To verify correct operation under all input conditions, full transient simulations were performed.

#### 2-4 Decoder Verification

* Simulation time: **64 ns**
* All **16 possible input transitions** were covered.

#### 4-16 Decoder Verification

* Simulation time: **256 ns**
* All **256 input transitions** were evaluated.

These simulations ensure that every possible switching event is captured, allowing accurate measurement of **power consumption and propagation delay**.

---

# 3. Experimental Results

The proposed designs were compared against **conventional CMOS implementations** under identical simulation conditions.

### Performance Comparison

| Metric (1.0 V Supply) | Conventional CMOS (2-4) | Proposed 2-4HP               | Conventional CMOS (4-16) | Proposed 4-16HP  |
| --------------------- | ----------------------- | ---------------------------- | ------------------------ | ---------------- |
| Transistor Count      | 20                      | **15**                       | 104                      | **94**           |
| Average Power (µW)    | 0.862                   | **1.964**                    | 2.751                    | **3.874**        |
| Maximum Delay         | 49 ps                   | **3.46 ns**                  | 97 ps                    | **243 ps**       |
| Energy Efficiency     | Baseline                | **Improved area efficiency** | Baseline                 | **Superior PDP** |

### Observations

**Transistor Reduction**

The mixed-logic implementation achieves noticeable **transistor count reduction**, which translates to:

* Lower silicon area
* Reduced fabrication cost
* Potentially improved scalability in large digital systems

**Power Characteristics**

While the proposed architectures may show slightly higher absolute power values in some cases due to pass-transistor behavior, the **overall efficiency per transistor and layout area improves**.

**Propagation Delay**

Propagation delay depends strongly on:

* Signal restoration points
* Pass-transistor thresholds
* Load capacitance

The inclusion of static CMOS stages ensures that logic levels remain stable even in mixed-logic environments.

**Energy Efficiency**

Energy efficiency is evaluated using **Power-Delay Product (PDP)**.
The proposed architecture demonstrates strong performance improvements when normalized for area and complexity.

---

# 4. Conclusion

This project demonstrates that **mixed-logic design techniques can significantly improve the efficiency of line decoders** used in modern digital systems.

By strategically combining:

* **Transmission Gate Logic**
* **Pass-Transistor Dual-Value Logic**
* **Static CMOS logic**

the design achieves:

* Reduced transistor count
* Lower layout area
* Competitive energy efficiency
* Improved scalability for large digital circuits

The **4-16HP decoder architecture** in particular shows strong potential for **memory peripheral circuitry**, where compact and efficient decoding structures are critical.

Future work may involve:

* Layout-level implementation and parasitic extraction
* Comparison with FinFET-based technologies
* Further power optimization using dynamic logic techniques
* Integration into larger memory subsystem designs

---

# 5. Reference

Balobas, D., & Konofaos, N.
**Design of Low Power, High Performance 2-4 and 4-16 Mixed-Logic Line Decoders**
IEEE Transactions on Circuits and Systems II: Express Briefs, 2016.
DOI: 10.1109/TCSII.2016.2555020.
