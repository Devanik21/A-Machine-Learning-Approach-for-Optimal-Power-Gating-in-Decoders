# ⚡ Low-Power Mixed-Logic Line Decoders — 2-to-4 HP & 4-to-16 HP
### Design, Simulation & Verification in LTspice | 32nm PTM LP Technology

> **Author:** Devanik Debnath  
> **GitHub:** [@Devanik21](https://github.com/Devanik21)  
> **LinkedIn:** [linkedin.com/in/devanik](https://www.linkedin.com/in/devanik/)  
> **Institution:** National Institute of Technology Agartala — ECE Department  
> **Academic Year:** 2025–2026  
> **Project Category:** VLSI Design · Low-Power Digital Circuits · Mixed-Logic Architectures

---

## 📋 Table of Contents

1. [Project Abstract](#1-project-abstract)
2. [Motivation & Background](#2-motivation--background)
3. [Technology Framework](#3-technology-framework)
4. [Mathematical Foundation](#4-mathematical-foundation)
5. [Logic Style Taxonomy](#5-logic-style-taxonomy)
   - [Transmission Gate Logic (TGL)](#51-transmission-gate-logic-tgl)
   - [Dual-Value Logic (DVL)](#52-dual-value-logic-dvl)
   - [Static CMOS](#53-static-cmos)
6. [Proposed 2-to-4 HP Mixed-Logic Decoder](#6-proposed-2-to-4-hp-mixed-logic-decoder)
   - [Architecture Overview](#61-architecture-overview)
   - [Transistor-Level Breakdown](#62-transistor-level-breakdown)
   - [Output Logic & Connections](#63-output-logic--connections)
   - [Block Diagram Description](#64-block-diagram-description)
   - [Signal Restoration Rationale](#65-signal-restoration-rationale)
7. [Proposed 4-to-16 HP Mixed-Logic Decoder](#7-proposed-4-to-16-hp-mixed-logic-decoder)
   - [Predecoding Architecture](#71-predecoding-architecture)
   - [Transistor-Level Breakdown](#72-transistor-level-breakdown)
   - [Block Diagram Description](#73-block-diagram-description)
8. [Power Analysis — Mathematical Treatment](#8-power-analysis--mathematical-treatment)
9. [Delay Analysis — Mathematical Treatment](#9-delay-analysis--mathematical-treatment)
10. [Power-Delay Product (PDP)](#10-power-delay-product-pdp)
11. [Simulation Results — 2-to-4 HP Decoder](#11-simulation-results--2-to-4-hp-decoder)
12. [Simulation Results — 4-to-16 HP Decoder](#12-simulation-results--4-to-16-hp-decoder)
13. [Comparative Analysis](#13-comparative-analysis)
14. [LTspice Implementation Guide](#14-ltspice-implementation-guide)
15. [Design Flowchart Logic](#15-design-flowchart-logic)
16. [Conclusion & Future Scope](#16-conclusion--future-scope)
17. [References](#17-references)

---

## 1. Project Abstract

This repository presents the complete design, simulation, and verification of **low-power mixed-logic line decoders** — specifically a **15-transistor 2-to-4 High-Performance (2-4HP)** decoder and a **94-transistor 4-to-16 High-Performance (4-16HP)** decoder — implemented using a **32nm Predictive Technology Model (PTM) for Low-Power (LP)** applications.

The core innovation lies in the deliberate departure from conventional **Static CMOS complementary logic** (which mandates 2N transistors for N-input logic) toward a **mixed-logic strategy** that synergistically integrates three paradigms:

- **Transmission Gate Logic (TGL)** — for pass-transistor-based efficient AND operations
- **Dual-Value Logic (DVL)** — for complementary pass-transistor paths
- **Static CMOS** — for signal restoration and logic integrity

The design achieves a **25% transistor area reduction** at the 2-to-4 level and a **24.3% average power reduction** at the 4-to-16 level (1.945 µW vs 2.572 µW benchmark), while maintaining **full-swing logic (0.0 V – 1.0 V)** across all output transitions. All 256 input transitions for the 4-to-16 decoder were verified through transient simulation in **LTspice**.

---

## 2. Motivation & Background

Modern VLSI systems — particularly **SRAM arrays, memory address decoders, and high-density logic fabrics** — place increasingly severe constraints on power consumption, area, and speed. Line decoders form a critical component in such systems, translating binary-encoded addresses into a single active output line.

### Why Conventional CMOS Falls Short

A conventional Static CMOS $N$-input gate requires:

$$T_{CMOS} = 2N \text{ transistors}$$

For a **2-to-4 decoder** (4 minterms, each needing complementary CMOS AND logic plus 2 inverters):

$$T_{conv, 2-4} = 4 \times 4 + 2 \times 2 = 20 \text{ transistors}$$

For a **4-to-16 decoder**:

$$T_{conv, 4-16} = 16 \times 4 + 4 \times 2 = 72 + 32 = 104 \text{ transistors}$$

While functionally correct, this imposes:
- High parasitic capacitance due to redundant complementary pairs
- Unnecessary switching activity on symmetric complementary nodes
- Greater area overhead that scales poorly with input size

The **mixed-logic approach** attacks all three simultaneously.

---

## 3. Technology Framework

| Parameter | Specification |
|-----------|--------------|
| **Simulation Tool** | LTspice (Analog Devices) |
| **Technology Node** | 32nm Predictive Technology Model (PTM) — Low Power (LP) |
| **Supply Voltage (V_DD)** | 1.0 V |
| **NMOS Model** | 32nm PTM LP nMOS |
| **PMOS Model** | 32nm PTM LP pMOS |
| **Input Transition Ramp** | 250 ps (realistic ramp — not ideal step) |
| **Simulation Type** | Transient Analysis |
| **Operating Frequency Range** | High-speed regime (verified at 191.5 ps max delay) |

> The **Predictive Technology Model (PTM)** is an industry-standard SPICE model developed at Arizona State University, calibrated to ITRS roadmap targets for advanced nodes. Using the LP (Low-Power) variant ensures that leakage-current characteristics reflect real-world standby power constraints at 32nm.

---

## 4. Mathematical Foundation

### 4.1 Decoder Boolean Logic

A binary **$n$-to-$2^n$ line decoder** implements the complete minterm set of $n$ variables. For inputs $A, B$ (2-to-4 decoder):

| Minterm | Boolean Expression | Output |
|---------|-------------------|--------|
| $m_0$ | $\bar{A} \cdot \bar{B}$ | $D_0$ |
| $m_1$ | $A \cdot \bar{B}$ | $D_1$ |
| $m_2$ | $\bar{A} \cdot B$ | $D_2$ |
| $m_3$ | $A \cdot B$ | $D_3$ |

**Constraint:** Exactly one output is HIGH for any given input combination:

$$\sum_{i=0}^{3} D_i = 1 \quad \forall \; (A, B) \in \{0,1\}^2$$

### 4.2 Total Power Dissipation Model

Total power in a CMOS circuit is expressed as:

$$P_{total} = P_{dynamic} + P_{short-circuit} + P_{leakage}$$

Where:

$$P_{dynamic} = \alpha \cdot C_{L} \cdot V_{DD}^2 \cdot f$$

$$P_{leakage} = I_{leakage} \cdot V_{DD}$$

- $\alpha$ = activity factor (switching probability per clock cycle)  
- $C_L$ = load capacitance at output node  
- $V_{DD}$ = supply voltage  
- $f$ = operating frequency  
- $I_{leakage}$ = sum of subthreshold and gate leakage currents

**Mixed-logic reduction mechanism:** By eliminating transistors, the total gate capacitance $C_L$ decreases, directly reducing $P_{dynamic}$.

### 4.3 Propagation Delay Model

For an RC chain (simplified Elmore delay):

$$t_p = 0.69 \sum_{i} R_i \cdot C_i$$

For a pass-transistor (TGL) path with on-resistance $R_{on}$:

$$R_{on} = \frac{1}{\mu_n C_{ox} \frac{W}{L} (V_{GS} - V_{th})}$$

At 32nm, $V_{th} \approx 0.28\text{V}$ (LP NMOS), and with $V_{DD} = 1.0\text{V}$, the overdrive $V_{GS} - V_{th} = 0.72\text{V}$ governs the pass-transistor drive strength.

---

## 5. Logic Style Taxonomy

### 5.1 Transmission Gate Logic (TGL)

A **Transmission Gate** consists of an NMOS and PMOS transistor connected in parallel, with complementary gate signals:

```
         Gate (G)
          |
In ──┤N├──┬──┤P├── Out
          |
        Gate̅ (G̅)
```

**AND gate using TGL (3 transistors):**

- Control signal $G$ drives NMOS gate; $\bar{G}$ drives PMOS gate
- Propagate signal feeds the source terminal
- Output = $G \cdot \text{Propagate}$

**Advantages:**
- Bidirectional, near-ideal switch
- $R_{on}$ remains low across full input range (both NMOS and PMOS conduct in complementary regions)
- 3 transistors vs 6 in CMOS AND

**Disadvantage:**
- Threshold voltage drop ($V_{th}$ loss) on NMOS-only paths → mitigated by using full TG (both N and P)

### 5.2 Dual-Value Logic (DVL)

DVL exploits the **dual nature** of Boolean expressions, using both true and complement forms of signals to drive separate transistor networks. A DVL AND gate (3 transistors) uses:

- The complement of the control signal ($\bar{A}$) to steer logic
- Direct input $B$ as the propagate signal

This eliminates the need for additional inversions and allows reuse of already-generated complemented signals.

**Key property:** DVL paths operate on signals that are **structurally complementary** within the same circuit, avoiding redundant inversions while maintaining logical correctness.

### 5.3 Static CMOS

Used selectively in this design for two critical purposes:

1. **Signal Restoration:** The $D_0$ NOR gate restores full-swing signals degraded by pass-transistor paths
2. **Input Stage:** A single CMOS inverter generates $\bar{A}$ from $A$

**Full-complementary CMOS NOR gate (4 transistors):**

$$D_0 = \overline{A + B} = \overline{A} \cdot \overline{B}$$

When inputs are $\bar{A}$ and $B$:

$$D_0 = \overline{\bar{A} + B} = A \cdot \bar{B} \quad \leftarrow \text{Wait — corrected form below}$$

Actually: $\text{NOR}(\bar{A}, B) = \overline{\bar{A} + B} = A \cdot \bar{B}$... 

The correct application: NOR gate with inputs $A_{in1} = \bar{A}$ and $A_{in2} = B$ computes:

$$D_0 = \overline{\bar{A} + B}$$

But $\bar{A} + B$ is NOT $\overline{\bar{A} \cdot \bar{B}}$... The design uses the identity:

$$\text{NOR}(A, B) = \overline{A + B} = \bar{A} \cdot \bar{B}$$

So with inputs $A$ and $B$ directly into the NOR gate:

$$D_0 = \text{NOR}(A, B) = \bar{A} \cdot \bar{B} \checkmark$$

This is the $m_0$ minterm — correct and full-swing restored.

---

## 6. Proposed 2-to-4 HP Mixed-Logic Decoder

### 6.1 Architecture Overview

The **2-4HP decoder** uses **15 transistors** (9 nMOS + 6 pMOS) compared to **20 transistors** in conventional CMOS — a **25% area reduction**.

The architecture is divided into two functional stages:

```
                    ┌─────────────────────────────────┐
  A ──────────┬─────┤                                 ├──► D₀ = Ā·B̄  (NOR restored)
              │     │   MIXED-LOGIC MINTERM           ├──► D₁ = A·B̄  (TGL AND)
  A ──[INV]───┤  Ā  │   GENERATION STAGE              ├──► D₂ = Ā·B  (DVL AND)
              │     │   (13 Transistors)              ├──► D₃ = A·B  (TGL AND)
  B ──────────┘     │                                 │
                    └─────────────────────────────────┘
  Input Stage (2T)
```

**Available signals after Input Stage:**
- $A$ (direct)
- $\bar{A}$ (inverted via 1 Static CMOS inverter = 2 transistors)
- $B$ (direct — **no inverter generated**)

**Key design decision:** $\bar{B}$ is **never explicitly generated**. Its effect is absorbed structurally into the gate connections of TGL and DVL gates.

### 6.2 Transistor-Level Breakdown

| Stage | Gate Type | Logic Function | Transistor Count | nMOS | pMOS |
|-------|-----------|---------------|-----------------|------|------|
| Input Stage | Static CMOS Inverter | Generate $\bar{A}$ | 2 | 1 | 1 |
| $D_0$ Output | Static CMOS NOR | $\bar{A} \cdot \bar{B}$ | 4 | 2 | 2 |
| $D_1$ Output | TGL AND | $A \cdot \bar{B}$ | 3 | 2 | 1 |
| $D_2$ Output | DVL AND | $\bar{A} \cdot B$ | 3 | 1 | 2 |
| $D_3$ Output | TGL AND | $A \cdot B$ | 3 | 2 | 1 |
| **TOTAL** | — | — | **15** | **9** | **6** |

### 6.3 Output Logic & Connections

#### Output $D_0$ — Static CMOS NOR Gate (Signal Restorer)

| Attribute | Detail |
|-----------|--------|
| Logical Function | $D_0 = \bar{A} \cdot \bar{B}$ |
| Gate Type | Static CMOS NOR (4 transistors: 2 nMOS in series pull-down, 2 pMOS in parallel pull-up) |
| Inputs | $A$ and $B$ (direct) |
| Operation | $\text{NOR}(A, B) = \overline{A + B} = \bar{A} \cdot \bar{B}$ |
| Role | Signal restorer — ensures full-swing (0 V to 1.0 V) at $D_0$, setting stable reference for decoder |

**Why NOR is used for $D_0$:** Pass-transistor paths (TGL/DVL) suffer from potential signal degradation at intermediate nodes due to $V_{th}$ drops. By anchoring $D_0$ as a full-CMOS NOR gate, the circuit maintains logic integrity across all outputs. $D_0$ itself benefits from the strong drive strength of CMOS push-pull topology.

#### Output $D_1$ — TGL AND Gate

| Attribute | Detail |
|-----------|--------|
| Logical Function | $D_1 = A \cdot \bar{B}$ |
| Gate Type | Transmission Gate Logic AND (3 transistors) |
| Control Signal | $A$ |
| Propagate Signal | $B$ |
| Structural Note | The absence of $\bar{B}$ is handled by wiring: when $A=1$, the gate passes $\bar{B}$ structurally through the asymmetric connection |

#### Output $D_2$ — DVL AND Gate

| Attribute | Detail |
|-----------|--------|
| Logical Function | $D_2 = \bar{A} \cdot B$ |
| Gate Type | Dual-Value Logic AND (3 transistors) |
| Control Signal | $\bar{A}$ |
| Propagate Signal | $B$ |
| Note | DVL uses the already-generated $\bar{A}$, eliminating any additional inversion overhead |

#### Output $D_3$ — TGL AND Gate

| Attribute | Detail |
|-----------|--------|
| Logical Function | $D_3 = A \cdot B$ |
| Gate Type | Transmission Gate Logic AND (3 transistors) |
| Control Signal | $A$ |
| Propagate Signal | $B$ |

### 6.4 Block Diagram Description

```
Input A ──────[INV 2T]──────► Ā
         │
         └──────────────────► A
Input B  ──────────────────► B   (no inversion — direct path)

Available Signals: {A, Ā, B}

                    ┌─────────────────────────────────────────────┐
         A ────────►│                                             │
         B ────────►│  Static CMOS NOR (4T)  ──────────────────►  D₀ = Ā·B̄
                    │                                             │
         A ────────►│                                             │
         B ────────►│  TGL AND (3T)          ──────────────────►  D₁ = A·B̄
                    │                                             │
         Ā ────────►│                                             │
         B ────────►│  DVL AND (3T)          ──────────────────►  D₂ = Ā·B
                    │                                             │
         A ────────►│                                             │
         B ────────►│  TGL AND (3T)          ──────────────────►  D₃ = A·B
                    └─────────────────────────────────────────────┘
                          Minterm Generation Stage (13T)
```

**Total: 2T (input) + 13T (minterms) = 15T**

### 6.5 Signal Restoration Rationale

In pass-transistor logic, an NMOS pass transistor can only pull a node up to $V_{DD} - V_{th,n}$ (approximately $1.0 - 0.28 = 0.72$ V at 32nm LP). This **threshold voltage loss** causes:

$$V_{out,high} = V_{DD} - V_{th,n} < V_{DD}$$

For subsequent stages, this degraded HIGH level can cause:
- Increased short-circuit current (both PMOS and NMOS partially on)
- Incorrect logic evaluation in chained gates
- Reduced noise margins

The **Static CMOS NOR at $D_0$** anchors the signal reference. Since the DVL/TGL gates for $D_1$–$D_3$ use both NMOS and PMOS (full transmission gates), they avoid this issue — but the NOR gate provides an architectural reference point that stabilizes the overall decoder operation.

---

## 7. Proposed 4-to-16 HP Mixed-Logic Decoder

### 7.1 Predecoding Architecture

The 4-to-16HP decoder employs a **two-stage predecoding strategy** — a classical but here elegantly optimized approach:

```
Inputs: A, B, C, D

Stage 1: PREDECODER (30T)
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  A, B ──► [ 2-4HP Mixed-Logic Decoder (15T) ] ──► X₀,X₁,X₂,X₃  │
│                                                                   │
│  C, D ──► [ 2-4HP Mixed-Logic Decoder (15T) ] ──► Y₀,Y₁,Y₂,Y₃  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                    8 intermediate signals
                    {X₀,X₁,X₂,X₃,Y₀,Y₁,Y₂,Y₃}
                              │
Stage 2: POST-DECODER (64T)
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  D_i = NOR(Xⱼ, Yₖ)  for all 16 combinations (i = 0..15)        │
│  Each NOR gate: 4 transistors (Static CMOS 2-input NOR)          │
│  Total: 16 × 4 = 64 transistors                                  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Why NOR for post-decoder?**

The intermediate signals $X_j$ and $Y_k$ from the predecoder are **active-HIGH** minterms. To generate $D_i = X_j \cdot Y_k$ (AND), we observe:

$$X_j \cdot Y_k = \overline{\bar{X_j} + \bar{Y_k}} = \overline{\overline{X_j \cdot Y_k}}$$

Since $X_j$ and $Y_k$ are minterms (only one HIGH at a time from each group), a NOR gate on the *active signals* effectively computes the AND of the two minterm sets. The polarity convention is embedded in the predecoder outputs being active-LOW in the NOR-based scheme, which the NOR gate naturally handles to produce active-HIGH final outputs.

### 7.2 Transistor-Level Breakdown

| Stage | Component | Count | Transistors Each | Total Transistors |
|-------|-----------|-------|-----------------|-------------------|
| Predecoder | 2-4HP Decoder (A,B) | 1 | 15 | 15 |
| Predecoder | 2-4HP Decoder (C,D) | 1 | 15 | 15 |
| Post-Decoder | Static CMOS 2-input NOR | 16 | 4 | 64 |
| **TOTAL** | | | | **94** |

**Conventional CMOS equivalent:** 104 transistors  
**Reduction:** $\frac{104 - 94}{104} \times 100 = 9.6\%$ transistor count reduction

### 7.3 Block Diagram Description

```
         ┌─────────────────┐
A ──────►│  2-to-4 HP      │──► X₀ = Ā·B̄
B ──────►│  Mixed-Logic     │──► X₁ = A·B̄
         │  Decoder (15T)  │──► X₂ = Ā·B
         │  [Inputs: A, B] │──► X₃ = A·B
         └─────────────────┘           │
                                        │ 8 intermediate
         ┌─────────────────┐           │ signals fed to
C ──────►│  2-to-4 HP      │──► Y₀ = C̄·D̄   │ NOR post-decoder
D ──────►│  Mixed-Logic     │──► Y₁ = C·D̄   │
         │  Decoder (15T)  │──► Y₂ = C̄·D    │
         │  [Inputs: C, D] │──► Y₃ = C·D    │
         └─────────────────┘           │
                                        ▼
         ┌─────────────────────────────────────────┐
         │      POST-DECODER: 16× Static NOR (4T)  │
         │                                         │
         │  D₀  = NOR(X₀, Y₀) = X₀·Y₀ = Ā·B̄·C̄·D̄  │
         │  D₁  = NOR(X₁, Y₀) = A·B̄·C̄·D̄           │
         │  ...                                    │
         │  D₁₅ = NOR(X₃, Y₃) = A·B·C·D          │
         └─────────────────────────────────────────┘
                          │
              D₀, D₁, D₂, ..., D₁₅ (16 outputs)
```

**Fan-in Reduction Advantage:**

Conventional 4-to-16 CMOS would require 4-input AND gates (each with 8 transistors) for every minterm. With predecoding:

- Each NOR gate has **fan-in = 2** (instead of 4)
- Lower fan-in → smaller transistors → **reduced capacitance → lower dynamic power**

$$C_{gate, NOR2} \ll C_{gate, AND4}$$

This is the central power-reduction mechanism of the predecoding architecture.

---

## 8. Power Analysis — Mathematical Treatment

### 8.1 Dynamic Power Derivation

$$P_{dynamic} = \alpha \cdot C_L \cdot V_{DD}^2 \cdot f$$

For the 4-16HP decoder:

- $V_{DD} = 1.0$ V
- $P_{avg,\text{measured}} = 1.945\,\mu\text{W}$
- $P_{avg,\text{conventional}} = 2.572\,\mu\text{W}$

**Power savings:**

$$\Delta P = 2.572 - 1.945 = 0.627\,\mu\text{W}$$

$$\text{Power Reduction} = \frac{\Delta P}{P_{conv}} \times 100 = \frac{0.627}{2.572} \times 100 \approx 24.38\%$$

### 8.2 Capacitance Reduction Model

Since the mixed-logic decoder eliminates **10 transistors** from the conventional design (94 vs 104), the gate capacitance per eliminated transistor at 32nm PTM LP is approximately:

$$C_{gate,32nm} \approx C_{ox} \cdot W \cdot L$$

With minimum-width transistors ($W_{min} = 3\lambda$, $L = 32\