# Domain Study: Finite-Window Shadowing and Identifiability Under Sparse Observations

## Abstract

This repository studies finite-window shadowing for linear hyperbolic automorphisms of the two-torus under intentionally sparse observations. In this setting, explicit stable and unstable decompositions yield constructive finite-window shadowing bounds with computable constants. The observation operator, restricted to a single coordinate and applied only at the window endpoints, induces an arithmetic non-injectivity mechanism that yields multiple distinct initial states consistent with identical observations. The resulting inverse problem is set-valued even when the model defect is identically zero. The repository implements the theoretical constructions and reproduces the numerical experiments that illustrate both the shadowing guarantees and the identifiability obstruction.

---

## 1. Problem statement and objectives

Let $F:\mathbb{T}^2\to\mathbb{T}^2$ be a hyperbolic toral automorphism. Given a finite sequence $(x_0, x_1, \ldots, x_q)$ (a candidate trajectory) and an observation schedule consisting of a single coordinate observed at the endpoints, two mathematically distinct questions arise:

1. **Finite-window shadowing (existence):** If the defect $r_k = x_{k+1} - F(x_k)$ is small, does there exist an exact orbit $(y_0, y_1, \ldots, y_q)$ of $F$ remaining uniformly close to $(x_0, x_1, \ldots, x_q)$ on the window?
2. **Identifiability (uniqueness):** Given endpoint observations $(H(x_0), H(x_q))$, is the compatible initial condition $x_0$ uniquely determined?

This study provides:

- An explicit finite-window shadowing certificate derived from the hyperbolic splitting and oblique projectors.
- An arithmetic characterization of the endpoint-only, single-coordinate inverse map, including explicit formulas for the family of indistinguishable initial states.
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

Let $A \in SL(2,\mathbb{Z})$ be hyperbolic. It has real eigenvalues $\lambda_u,\lambda_s$ with

$$
|\lambda_u|>1,\qquad |\lambda_s|<1,\qquad \lambda_u\lambda_s=1.
$$

Define the map

$$
F:\mathbb{T}^2 \to \mathbb{T}^2,\qquad F(x)=Ax \bmod 1.
$$

For an integer $q\ge 1$, denote

$$
A^q=
\begin{pmatrix}
 a_q & b_q\\
 c_q & d_q
\end{pmatrix}.
$$

The entry $b_q$ is central in the identifiability analysis.

---

## 3. Hyperbolic splitting and oblique projectors

### 3.1. Spectral decomposition

Since $A$ is hyperbolic, $\mathbb{R}^2$ decomposes as

$$
\mathbb{R}^2 = E^u \oplus E^s,
$$

where $E^u$ and $E^s$ are the unstable and stable eigenspaces.

Let $v_u \in E^u$ and $v_s \in E^s$ be eigenvectors and define

$$
Q := [\,v_u\ \ v_s\,],\qquad \Lambda := \mathrm{diag}(\lambda_u,\lambda_s),
$$

so that $A = Q\Lambda Q^{-1}$.

### 3.2. Oblique projectors

Define the spectral projectors

$$
P_u := Q
\begin{pmatrix}
1 & 0\\
0 & 0
\end{pmatrix}
Q^{-1},
\qquad
P_s := Q
\begin{pmatrix}
0 & 0\\
0 & 1
\end{pmatrix}
Q^{-1}.
$$

These satisfy

$$
P_u + P_s = I,\qquad P_u^2=P_u,\quad P_s^2=P_s,
$$

and commute with $A$:

$$
AP_u=P_uA,\qquad AP_s=P_sA.
$$

Every $z\in\mathbb{R}^2$ decomposes uniquely as $z=z^u+z^s$ with

$$
z^u:=P_uz\in E^u,\qquad z^s:=P_sz\in E^s.
$$

### 3.3. Projection norm in dimension two

Let $\theta\in(0,\pi)$ denote the angle between the unit eigenvectors spanning $E^u$ and $E^s$ (with respect to the Euclidean inner product). In dimension two, the oblique projector norms satisfy

$$
\|P_u\|_2=\|P_s\|_2=\frac{1}{|\sin\theta|}.
$$

This constant quantifies the conditioning of the stable and unstable splitting and enters explicitly in shadowing constants.

---

## 4. Finite-window pseudo-orbits and defect

Let $(x_0, x_1, \ldots, x_q) \subset \mathbb{T}^2$ be a finite sequence. Choose a lift $(x_0, x_1, \ldots, x_q) \subset \mathbb{R}^2$ and use the same symbols for lifted representatives. Fix the lift convention so that the defect is represented canonically: for each $k$ there exists an integer vector $n_k\in\mathbb{Z}^2$ such that

$$
r_k := x_{k+1} - A x_k - n_k \in [-1/2,1/2)^2,\qquad k=0,\dots,q-1.
$$

Equivalently, $r_k$ is the canonical representative of the torus difference $x_{k+1}-F(x_k)$.

A uniform defect bound has the form

$$
\|r_k\|_2\le \varepsilon,\qquad k=0,\dots,q-1.
$$

---

## 5. Finite-window shadowing as a linear boundary-value problem

### 5.1. Correction recurrence

An exact orbit lift $(y_0, y_1, \ldots, y_q)\subset\mathbb{R}^2$ satisfies $y_{k+1}=Ay_k+n_k$ for the same integer sequence $(n_0, n_1, \ldots, n_{q-1})$ used above. Writing $y_k=x_k+d_k$ yields the inhomogeneous recurrence

$$
d_{k+1} = A d_k - r_k,\qquad k=0,\dots,q-1.
$$

### 5.2. Hyperbolic boundary constraints

Over a finite window, the recurrence admits multiple solutions unless boundary conditions are imposed. A standard hyperbolic choice is

$$
P_s d_0 = 0,\qquad P_u d_q = 0.
$$

These conditions define a two-point boundary-value problem associated with the stable and unstable splitting.

### 5.3. Componentwise solution formulas

Projecting the recurrence and using commutation with $A$ gives

$$
d_{k+1}^u = A d_k^u - r_k^u,\qquad r_k^u:=P_u r_k,
$$

$$
d_{k+1}^s = A d_k^s - r_k^s,\qquad r_k^s:=P_s r_k.
$$

Using $d_q^u=0$, solve the unstable component backward:

$$
d_k^u = \sum_{j=k}^{q-1} A^{-(j-k+1)} r_j^u.
$$

Using $d_0^s=0$, solve the stable component forward:

$$
d_k^s = -\sum_{j=0}^{k-1} A^{k-1-j} r_j^s.
$$

These representations are the basis for the finite-window bounds.

---

## 6. Explicit finite-window shadowing certificate

### 6.1. Geometric-series bound

Assume $\|r_k\|_2\le\varepsilon$ for all $k$. Then

$$
\|r_k^u\|_2\le \|P_u\|_2\,\varepsilon,\qquad \|r_k^s\|_2\le \|P_s\|_2\,\varepsilon.
$$

On $E^u$, $A^{-m}$ contracts with rate $|\lambda_u|^{-m}$; on $E^s$, $A^{m}$ contracts with rate $|\lambda_s|^{m}$. Bounding the series yields

$$
\max_{0\le k\le q}\|d_k\|_2\le C(q;A)\,\varepsilon,
$$

with an explicit admissible choice

$$
C(q;A)=
\|P_u\|_2\sum_{m=1}^{q}|\lambda_u|^{-m}
+\|P_s\|_2\sum_{m=0}^{q-1}|\lambda_s|^{m}.
$$

Using closed forms,

$$
\sum_{m=1}^{q}|\lambda_u|^{-m}
=
\frac{|\lambda_u|^{-1}(1-|\lambda_u|^{-q})}{1-|\lambda_u|^{-1}},
\qquad
\sum_{m=0}^{q-1}|\lambda_s|^{m}
=
\frac{1-|\lambda_s|^{q}}{1-|\lambda_s|}.
$$

### 6.2. Interpretation

The constant $C(q;A)$ is computable from:

- the unstable and stable rates $|\lambda_u|,|\lambda_s|$,
- the obliquity factor $\|P_u\|_2=\|P_s\|_2=1/|\sin\theta|$,
- the window length $q$.

---

## 7. Observation model

### 7.1. Observation operator and schedule

Define the observation operator

$$
H:\mathbb{T}^2 \to \mathbb{T},\qquad H(x_1,x_2)=x_1 \bmod 1.
$$

The observation schedule is endpoint-only:

$$
y_0 = H(x_0),\qquad y_q = H(x_q).
$$

### 7.2. Observation map

Define the endpoint observation map

$$
\mathcal{G}_q:\mathbb{T}^2 \to \mathbb{T}\times\mathbb{T},\qquad \mathcal{G}_q(x_0)=(H(x_0),H(F^q(x_0))).
$$

Identifiability corresponds to injectivity of $\mathcal{G}_q$.

---

## 8. Arithmetic characterization of the indistinguishable set

Let $x_0=(x_{0,1},x_{0,2})\in\mathbb{R}^2$ be a lift, so $x_q=A^q x_0$ (as a lift). Using

$$
A^q=
\begin{pmatrix}
 a_q & b_q\\
 c_q & d_q
\end{pmatrix},
$$

the endpoint observations impose

$$
x_{0,1}\equiv y_0 \pmod{1},
$$

$$
a_q x_{0,1}+b_q x_{0,2}\equiv y_q \pmod{1}.
$$

Substituting $x_{0,1}\equiv y_0$ yields the congruence

$$
b_q x_{0,2} \equiv y_q - a_q y_0 \pmod{1}.
$$

Assume $b_q\neq 0$. Then the solution set for $x_{0,2}$ modulo 1 is

$$
x_{0,2}^{(m)} \equiv \frac{y_q-a_q y_0+m}{b_q}\pmod{1},
\qquad m=0,1,\dots,|b_q|-1.
$$

If $b_q=0$, the endpoint constraint reduces to $y_q \equiv a_q y_0 \pmod 1$. Under this compatibility, $x_{0,2}$ is unconstrained and the preimage contains a one-dimensional continuum; under incompatibility, the preimage is empty.

---

## 9. A minimax lower bound induced by non-injectivity

### 9.1. Estimation model

Let $\widehat{x}_0:\mathbb{T}\times\mathbb{T}\to\mathbb{T}^2$ be any estimator mapping endpoint observations to an initial condition estimate. Let $\mathrm{dist}$ be the standard torus metric induced by Euclidean wrapping:

$$
\mathrm{dist}(x,x') := \|\mathrm{wrap}(x-x')\|_2,
$$

where $\mathrm{wrap}$ maps each coordinate into $[-1/2,1/2)$.

### 9.2. Lower bound from branch separation

Fix $(y_0,y_q)$ such that $|b_q|\ge 2$ and consider two adjacent branches $x_0^{(m)}$ and $x_0^{(m+1)}$ in the preimage set described above. These satisfy

$$
\mathcal{G}_q(x_0^{(m)})=\mathcal{G}_q(x_0^{(m+1)})=(y_0,y_q),
$$

and their wrapped separation is

$$
\mathrm{dist}(x_0^{(m)},x_0^{(m+1)})=\frac{1}{|b_q|}.
$$

By the two-point minimax argument,

$$
\max \lbrace
\mathrm{dist}(\widehat{x}_0(y_0,y_q),x_0^{(m)}),
\mathrm{dist}(\widehat{x}_0(y_0,y_q),x_0^{(m+1)})
\rbrace
\ge \frac{1}{2|b_q|}.
$$

Consequently,

$$
\inf_{\widehat{x}_0}\ \sup_{x_0:\ \mathcal{G}_q(x_0)=(y_0,y_q)}
\mathrm{dist}(\widehat{x}_0(y_0,y_q),x_0)
\ge \frac{1}{2|b_q|},
\qquad (|b_q|\ge 2).
$$

---

## 10. Asymptotics of the branch count via $|b_q|$

Since $A$ is diagonalizable over $\mathbb{R}$,

$$
A^q = Q \Lambda^q Q^{-1}.
$$

Each entry of $A^q$ is a linear combination of $\lambda_u^q$ and $\lambda_s^q$. In particular,

$$
b_q = \alpha \lambda_u^q + \beta \lambda_s^q
$$

for constants $\alpha,\beta$ determined by $A$.

Because $|\lambda_s|<1$, the stable contribution decays. Since $b_q$ is an integer sequence and $b_q\not\equiv 0$ for hyperbolic $A\in SL(2,\mathbb{Z})$, it follows that $\alpha\neq 0$. Therefore there exist constants $c_1,c_2>0$ and $q_0$ such that, for all $q\ge q_0$,

$$
c_1|\lambda_u|^q \le |b_q| \le c_2|\lambda_u|^q.
$$

Equivalently,

$$
\lim_{q\to\infty}\frac{1}{q}\log|b_q| = \log|\lambda_u|.
$$

Thus the cardinality of the indistinguishable preimage set grows exponentially with the window length $q$ for typical hyperbolic matrices.

---

## 11. Computational realization in this repository

The numerical component instantiates the preceding constructions in a reproducible workflow:

- Selection of representative hyperbolic matrices $A\in SL(2,\mathbb{Z})$.
- Construction of pseudo-orbits and exact-orbit corrections over finite windows using the boundary-value formulation.
- Verification of the explicit shadowing bound by comparing computed corrections to the predicted certificate scale.
- Branch discovery and auditing for the endpoint observation map via explicit congruence constraints associated with $b_q$.
- Empirical evaluation of how restart-based branch discovery saturates relative to the theoretically predicted branch count.

The computations are deterministic conditional on the seeds used for restart initialization. Random restarts are used to locate multiple solutions of a fixed finite-dimensional constraint system.

---

## 12. Relation to observability and symbolic dynamics (context)

The endpoint observation map $\mathcal{G}_q$ is a factor map from the state space to an observation space. In linear settings, identifiability relates to observability under restricted measurement operators. In uniformly hyperbolic dynamics, partial observations admit descriptions in terms of induced symbolic factors associated with Markov partitions, with non-injectivity corresponding to multiple symbolic sequences sharing the same observed factor sequence. The present setting isolates the mechanism in a linear toral model where the non-injectivity is controlled by matrix entries of $A^q$.

---

## 13. Scope and extensions

### 13.1. Scope

The study is restricted to:

- two-dimensional hyperbolic toral automorphisms,
- endpoint-only observation schedules,
- a single observed coordinate.

### 13.2. Extensions

Natural extensions include:

- observation schedules with intermediate times,
- observation of multiple coordinates,
- higher-dimensional hyperbolic toral automorphisms in $SL(d,\mathbb{Z})$,
- nonlinear Anosov diffeomorphisms, where branch structure may be studied via symbolic dynamics and factor maps.

---

## 14. Further reading

- Hyperbolic dynamics and Anosov diffeomorphisms: Bowen; Katok and Hasselblatt.
- Shadowing theory: Pilyugin.
- Symbolic dynamics and coding: Lind and Marcus.
- Linear observability (contextual): Kalman; standard texts on state-space systems.
