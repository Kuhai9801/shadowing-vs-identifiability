# Domain Study: Finite-Window Shadowing and Identifiability Under Sparse Observations

## Abstract

This repository studies finite-window shadowing for linear hyperbolic automorphisms of the two-torus under intentionally sparse observations. In this setting, explicit stable and unstable decompositions yield constructive finite-window shadowing bounds with computable constants. In parallel, the observation operator, restricted to a single coordinate and applied only at the window endpoints, induces an arithmetic non-injectivity mechanism that produces a multiplicity of distinct initial states consistent with identical observations. The resulting inverse problem is set-valued even when the model defect is identically zero. The repository implements the theoretical constructions and reproduces the numerical experiments that illustrate both the shadowing guarantees and the identifiability obstruction.

---

## 1. Problem statement and objectives

Let $F:\mathbb{T}^2\to\mathbb{T}^2$ be a hyperbolic toral automorphism. Given a finite sequence $(x\_k)\_{k=0}^q$ (a candidate trajectory) and an observation schedule consisting of a single coordinate observed at the endpoints, two mathematically distinct questions arise:

1. **Finite-window shadowing (existence):** If the defect $r\_k = x\_{k+1} - F(x\_k)$ is small, does there exist an exact orbit $(y\_k)\_{k=0}^q$ of $F$ remaining uniformly close to $(x\_k)\_{k=0}^q$ on the window?
2. **Identifiability (uniqueness):** Given endpoint observations $(H(x\_0), H(x\_q))$, is the compatible initial condition $x\_0$ uniquely determined?

This study provides:

- An explicit finite-window shadowing certificate derived from the hyperbolic splitting and oblique projectors.
- A complete arithmetic characterization of the endpoint-only, single-coordinate inverse map, including explicit formulas for the family of indistinguishable initial states.
- A minimax lower bound demonstrating that any single-valued estimator incurs nontrivial worst-case error due to observation non-injectivity.

---

## 2. Mathematical setting

### 2.1. State space

The two-torus is defined by

$$
\mathbb{T}^2 := \mathbb{R}^2 / \mathbb{Z}^2,
$$

so $x \in \mathbb{T}^2$ is an equivalence class of $x \in \mathbb{R}^2$ under integer translations.

### 2.2. Dynamics

Let $A \in SL(2,\mathbb{Z})$ with $\det(A)=1$, and assume $A$ is hyperbolic: it has real eigenvalues $\lambda\_u,\lambda\_s$ with

$$
|\lambda\_u|>1,\qquad |\lambda\_s|<1,\qquad \lambda\_u\lambda\_s=1.
$$

Define the map

$$
F:\mathbb{T}^2 \to \mathbb{T}^2,\qquad F(x)=Ax \bmod 1.
$$

For an integer $q\ge 1$, denote

$$
A^q=
\begin{pmatrix}
 a\_q & b\_q\\
 c\_q & d\_q
\end{pmatrix}.
$$

The entry $b\_q$ is central in the identifiability analysis.

---

## 3. Hyperbolic splitting and oblique projectors

### 3.1. Spectral decomposition

Since $A$ is hyperbolic, $\mathbb{R}^2$ decomposes as

$$
\mathbb{R}^2 = E^u \oplus E^s,
$$

where $E^u$ and $E^s$ are the unstable and stable eigenspaces.

Let $v\_u \in E^u$ and $v\_s \in E^s$ be eigenvectors and define

$$
Q := [\,v\_u\ \ v\_s\,],\qquad \Lambda := \mathrm{diag}(\lambda\_u,\lambda\_s),
$$

so that $A = Q\Lambda Q^{-1}$.

### 3.2. Oblique projectors

Define the spectral projectors

$$
P\_u := Q
\begin{pmatrix}
1 & 0\\
0 & 0
\end{pmatrix}
Q^{-1},
\qquad
P\_s := Q
\begin{pmatrix}
0 & 0\\
0 & 1
\end{pmatrix}
Q^{-1}.
$$

These satisfy

$$
P\_u + P\_s = I,\qquad P\_u^2=P\_u,\quad P\_s^2=P\_s,
$$

and commute with $A$:

$$
AP\_u=P\_uA,\qquad AP\_s=P\_sA.
$$

Every $z\in\mathbb{R}^2$ decomposes uniquely as $z=z^u+z^s$ with

$$
z^u:=P\_uz\in E^u,\qquad z^s:=P\_sz\in E^s.
$$

### 3.3. Projection norm in dimension two

Let $\theta\in(0,\pi)$ denote the angle between the unit eigenvectors spanning $E^u$ and $E^s$ (with respect to the Euclidean inner product). In dimension two, the oblique projector norms satisfy

$$
\|P\_u\|\_2=\|P\_s\|\_2=\frac{1}{|\sin\theta|}.
$$

This constant quantifies the conditioning of the stable and unstable splitting and enters explicitly in shadowing constants.

---

## 4. Finite-window pseudo-orbits and defect

Let $(x\_k)\_{k=0}^q \subset \mathbb{T}^2$ be a finite sequence. Choose a lift $(x\_k)\_{k=0}^q \subset \mathbb{R}^2$. Define the defect

$$
r\_k := x\_{k+1} - A x\_k,\qquad k=0,\dots,q-1.
$$

A uniform defect bound has the form

$$
\|r\_k\|\le \varepsilon,\qquad k=0,\dots,q-1,
$$

for a chosen norm $\|\cdot\|$ on $\mathbb{R}^2$ (typically Euclidean).

---

## 5. Finite-window shadowing as a linear boundary-value problem

### 5.1. Correction recurrence

An exact orbit lift $(y\_k)\_{k=0}^q\subset\mathbb{R}^2$ satisfies $y\_{k+1}=Ay\_k$. Writing $y\_k=x\_k+d\_k$ yields the inhomogeneous recurrence

$$
d\_{k+1} = A d\_k - r\_k,\qquad k=0,\dots,q-1.
$$

### 5.2. Hyperbolic boundary constraints

Over a finite window, the recurrence admits multiple solutions unless boundary conditions are imposed. A standard hyperbolic choice is

$$
P\_s d\_0 = 0,\qquad P\_u d\_q = 0.
$$

These conditions define a well-posed two-point boundary-value problem because:
- stable components are controlled forward in time under $A$,
- unstable components are controlled backward in time under $A^{-1}$.

### 5.3. Componentwise solution formulas

Projecting the recurrence and using commutation with $A$ gives

$$
d\_{k+1}^u = A d\_k^u - r\_k^u,\qquad r\_k^u:=P\_u r\_k,
$$

$$
d\_{k+1}^s = A d\_k^s - r\_k^s,\qquad r\_k^s:=P\_s r\_k.
$$

Using $d\_q^u=0$, solve the unstable component backward:

$$
d\_k^u = \sum\_{j=k}^{q-1} A^{k-j-1} r\_j^u.
$$

Using $d\_0^s=0$, solve the stable component forward:

$$
d\_k^s = -\sum\_{j=0}^{k-1} A^{k-1-j} r\_j^s.
$$

These explicit representations are the basis for the finite-window bounds.

---

## 6. Explicit finite-window shadowing certificate

### 6.1. Geometric-series bound

Assume $\|r\_k\|\le\varepsilon$ for all $k$. Then

$$
\|r\_k^u\|\le \|P\_u\|\,\varepsilon,\qquad \|r\_k^s\|\le \|P\_s\|\,\varepsilon.
$$

On $E^u$, $A^{-m}$ contracts with rate $|\lambda\_u|^{-m}$; on $E^s$, $A^{m}$ contracts with rate $|\lambda\_s|^{m}$. Bounding the series yields

$$
\max\_{0\le k\le q}\|d\_k\|\le C(q;A)\,\varepsilon,
$$

with an explicit admissible choice

$$
C(q;A)=
\|P\_u\|\sum\_{m=1}^{q}|\lambda\_u|^{-m}
+\|P\_s\|\sum\_{m=0}^{q-1}|\lambda\_s|^{m}.
$$

Using closed forms,

```math
\sum\_{m=1}^{q}|\lambda\_u|^{-m}
=
\frac{|\lambda\_u|^{-1}(1-|\lambda\_u|^{-q})}{1-|\lambda\_u|^{-1}},
\qquad
\sum\_{m=0}^{q-1}|\lambda\_s|^{m}
=
\frac{1-|\lambda\_s|^{q}}{1-|\lambda\_s|}.
```

### 6.2. Interpretation

The constant $C(q;A)$ is computable from:
- the unstable and stable rates $|\lambda\_u|,|\lambda\_s|$,
- the obliquity factor $\|P\_u\|=\|P\_s\|=1/|\sin\theta|$,
- the window length $q$.

The bound supplies a quantitative finite-window shadowing guarantee for any pseudo-orbit with uniformly bounded defect.

---

## 7. Observation model

### 7.1. Observation operator and schedule

Define the observation operator

$$
H:\mathbb{T}^2 \to \mathbb{T},\qquad H(x\_1,x\_2)=x\_1 \bmod 1.
$$

The observation schedule is endpoint-only:

$$
y\_0 = H(x\_0),\qquad y\_q = H(x\_q).
$$

### 7.2. Observation map

Define the endpoint observation map

$$
\mathcal{G}\_q:\mathbb{T}^2 \to \mathbb{T}\times\mathbb{T},\qquad \mathcal{G}\_q(x\_0)=(H(x\_0),H(F^q(x\_0))).
$$

Identifiability corresponds to injectivity of $\mathcal{G}\_q$.

---

## 8. Arithmetic characterization of the indistinguishable set

Let $x\_0=(x\_{0,1},x\_{0,2})\in\mathbb{R}^2$ be a lift, so $x\_q=A^q x\_0$. Using

$$
A^q=
\begin{pmatrix}
 a\_q & b\_q\\
 c\_q & d\_q
\end{pmatrix},
$$

the endpoint observations impose

$$
x\_{0,1}\equiv y\_0 \pmod{1},
$$

$$
a\_q x\_{0,1}+b\_q x\_{0,2}\equiv y\_q \pmod{1}.
$$

Substituting $x\_{0,1}\equiv y\_0$ yields the congruence

$$
b\_q x\_{0,2} \equiv y\_q - a\_q y\_0 \pmod{1}.
$$

Assume $b\_q\neq 0$. Then the solution set for $x\_{0,2}$ modulo 1 is

$$
x\_{0,2}^{(m)} \equiv \frac{y\_q-a\_q y\_0+m}{b\_q}\pmod{1},
\qquad m=0,1,\dots,|b\_q|-1.
$$

Consequently, for typical hyperbolic $A$ and typical observations, the preimage $\mathcal{G}\_q^{-1}(y\_0,y\_q)$ contains $|b\_q|$ distinct points in $\mathbb{T}^2$. The inverse problem defined by $(y\_0,y\_q)$ is set-valued.

---

## 9. A minimax lower bound induced by non-injectivity

### 9.1. Estimation model

Let $\widehat{x}\_0:\mathbb{T}\times\mathbb{T}\to\mathbb{T}^2$ be any estimator mapping endpoint observations to an initial condition estimate. Let $\mathrm{dist}$ be a metric on $\mathbb{T}^2$ compatible with the quotient topology (for example, the Euclidean distance on $[-1/2,1/2)^2$ after wrapping).

### 9.2. Lower bound from branch separation

Fix $(y\_0,y\_q)$ such that $|b\_q|\ge 2$ and consider two distinct branches $x\_0^{(m\_1)}$ and $x\_0^{(m\_2)}$ in the preimage set described above. These two states satisfy

$$
\mathcal{G}\_q(x\_0^{(m\_1)})=\mathcal{G}\_q(x\_0^{(m\_2)})=(y\_0,y\_q),
$$

yet their separation in the unobserved coordinate is on the order of $1/|b\_q|$.

By a standard two-point argument, for any estimator $\widehat{x}\_0$,

```math
\max \left\{
\mathrm{dist}\left(\widehat{x}\_0(y\_0,y\_q),x\_0^{(m\_1)}\right),
\mathrm{dist}\left(\widehat{x}\_0(y\_0,y\_q),x\_0^{(m\_2)}\right)
\right\}
\ge \frac{c}{|b\_q|},
```

for a metric-dependent constant $c>0$. Taking the supremum over admissible branches and infimum over estimators yields

```math
\inf\_{\widehat{x}\_0}\ \sup\_{x\_0:\ \mathcal{G}\_q(x\_0)=(y\_0,y\_q)}
\mathrm{dist}\left(\widehat{x}\_0(y\_0,y\_q),x\_0\right)
\ge \frac{c}{|b\_q|}.
```

The scaling $|b\_q|^{-1}$ is the salient feature: it follows directly from the explicit arithmetic structure of the preimage set.

---

## 10. Asymptotics of the branch count via $|b\_q|$

Since $A$ is diagonalizable over $\mathbb{R}$,

$$
A^q = Q \Lambda^q Q^{-1}.
$$

Each entry of $A^q$ is a linear combination of $\lambda\_u^q$ and $\lambda\_s^q$. In particular,

$$
b\_q = \alpha \lambda\_u^q + \beta \lambda\_s^q
$$

for constants $\alpha,\beta$ determined by $A$.

For hyperbolic $A$, one has $|\lambda\_s|<1$ and generically $\alpha\neq 0$, so there exist constants $c\_1,c\_2>0$ and $q\_0$ such that, for all $q\ge q\_0$,

$$
c\_1|\lambda\_u|^q \le |b\_q| \le c\_2|\lambda\_u|^q.
$$

Equivalently,

$$
\lim\_{q\to\infty}\frac{1}{q}\log|b\_q| = \log|\lambda\_u|.
$$

Thus the cardinality of the indistinguishable preimage set grows exponentially with the window length $q$ for typical hyperbolic matrices.

---

## 11. Computational realization in this repository

The numerical component instantiates the preceding constructions in a reproducible workflow:

- Selection of representative hyperbolic matrices $A\in SL(2,\mathbb{Z})$.
- Construction of pseudo-orbits and exact-orbit corrections over finite windows using the boundary-value formulation.
- Verification of the explicit shadowing bound by comparing computed corrections to the predicted certificate scale.
- Branch discovery and auditing for the endpoint observation map via explicit congruence constraints associated with $b\_q$.
- Empirical evaluation of how restart-based branch discovery saturates relative to the theoretically predicted branch count.

The computations are deterministic conditional on the chosen seeds used for restart initialization. Random restarts serve as a practical mechanism to locate multiple solutions of a fixed finite-dimensional constraint system; the underlying problem remains deterministic and algebraic.

---

## 12. Relation to observability and symbolic dynamics (context)

The endpoint observation map $\mathcal{G}\_q$ can be interpreted as a factor map from the full state space to a reduced observation space. In linear settings, identifiability is closely related to classical observability of state-space systems under restricted measurement operators. In uniformly hyperbolic dynamics, partial observations admit descriptions in terms of induced symbolic factors associated with Markov partitions, with non-injectivity manifesting as multiple symbolic sequences sharing the same observed factor sequence. The present setting isolates the mechanism in a linear toral model where the non-injectivity is explicitly controlled by matrix entries of $A^q$.

---

## 13. Scope and extensions

### 13.1. Scope

The study is restricted to:
- two-dimensional hyperbolic toral automorphisms,
- endpoint-only observation schedules,
- a single observed coordinate.

This restriction is methodological: it enables fully explicit constants, closed-form branch parametrizations, and transparent minimax statements.

### 13.2. Extensions

Natural extensions include:
- observation schedules with intermediate times,
- observation of multiple coordinates,
- higher-dimensional hyperbolic toral automorphisms in $SL(d,\mathbb{Z})$,
- nonlinear Anosov diffeomorphisms, where branch structure may be studied via symbolic dynamics and factor maps.

---

## 14. Notation

- $\mathbb{T}^2=\mathbb{R}^2/\mathbb{Z}^2$: two-torus.
- $A\in SL(2,\mathbb{Z})$: integer matrix with determinant 1.
- $\lambda\_u,\lambda\_s$: unstable and stable eigenvalues.
- $E^u,E^s$: unstable and stable eigenspaces.
- $P\_u,P\_s$: spectral oblique projectors onto $E^u$ and $E^s$.
- $\theta$: angle between unit eigenvectors spanning $E^u$ and $E^s$.
- $(x\_k)\_{k=0}^q$: lifted pseudo-orbit.
- $r\_k=x\_{k+1}-Ax\_k$: defect.
- $(d\_k)\_{k=0}^q$: correction sequence.
- $H(x\_1,x\_2)=x\_1\bmod 1$: observation operator.
- $\mathcal{G}\_q(x\_0)=(H(x\_0),H(F^q(x\_0)))$: endpoint observation map.
- $a\_q, b\_q, c\_q, d\_q$: entries of the iterate matrix $A^q$.

---

## 15. Further reading

- Hyperbolic dynamics and Anosov diffeomorphisms: Bowen; Katok and Hasselblatt.
- Shadowing theory: Pilyugin.
- Symbolic dynamics and coding: Lind and Marcus.
- Linear observability (contextual): Kalman; standard texts on state-space systems.
