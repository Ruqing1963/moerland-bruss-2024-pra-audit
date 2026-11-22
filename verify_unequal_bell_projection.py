#!/usr/bin/env python3
"""
verify_unequal_bell_projection.py

Deterministic verification script accompanying

  "Entanglement generation by unequal-dimension Bell-diagonal projections
   and a no-go theorem for local twirling"

which audits Proposition 1 of J. Moerland, N. Wyderka, H. Kampermann and D. Bruss,
Phys. Rev. A 109, 012217 (2024) [arXiv:2308.10607v2].  Equation numbers "MWKB (n)" refer to
the arXiv v2 numbering.

Conventions (identical to MWKB):
  H = C^{dA} (x) C^{dB}, dA <= dB, basis |i>_A|j>_B with index i*dB + j
  X_S|i> = |i+1 mod dS>,  Z_S|i> = w_S^i|i>,  w_S = exp(2 pi i/dS)
  Ztilde|i>_A = w_B^i|i>_A
  |phi^{00}> = dA^{-1/2} sum_{i<dA} |i>|i>,   |phi^{ab}> = (Z_A^a (x) X_B^b)|phi^{00}>
  T(mu,nu) = X_A^mu Ztilde^nu (x) X_B^mu Z_B^{-nu}
  Phi_1(rho) = (dA dB)^{-1} sum_{mu<dA, nu<dB} T(mu,nu) rho T(mu,nu)^dagger       [MWKB (19), q=1]
  Pi(rho)    = sum_{a,b} <phi^{ab}|rho|phi^{ab}> |phi^{ab}><phi^{ab}|            [r.h.s. of MWKB (20)]

Requires numpy, scipy, sympy.  Run:  python verify_unequal_bell_projection.py
Exit status 0 iff every check passes.  Runtime: a few minutes.
"""
import itertools
import math
import sys
import time
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.optimize import minimize

TOL = 1e-10
RESULTS = []
T0 = time.time()


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def header(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ----------------------------------------------------------------------------------------------
# basic operators
# ----------------------------------------------------------------------------------------------
def shift(d):
    X = np.zeros((d, d))
    for i in range(d):
        X[(i + 1) % d, i] = 1
    return X


def clock(d, n=None, root=None):
    """diag(root^i), i < n; default root = exp(2 pi i/d), n = d."""
    n = d if n is None else n
    root = np.exp(2j * np.pi / d) if root is None else root
    return np.diag(root ** np.arange(n))


def bell_basis(dA, dB):
    """Columns a*dB + b are |phi^{ab}>."""
    N = dA * dB
    wA = np.exp(2j * np.pi / dA)
    B = np.zeros((N, N), dtype=complex)
    for a in range(dA):
        for b in range(dB):
            for i in range(dA):
                B[i * dB + (i + b) % dB, a * dB + b] = wA ** (a * i) / np.sqrt(dA)
    return B


def op_S(dA, dB):
    """S = X_A (x) X_B = T(1,0)."""
    return np.kron(shift(dA), shift(dB))


def op_D(dA, dB):
    """D = Ztilde (x) Z_B^{-1} = T(0,1);  D|i,j> = w_B^{i-j}|i,j>."""
    wB = np.exp(2j * np.pi / dB)
    return np.kron(clock(dB, dA, wB), clock(dB, dB, wB).conj())


def op_C(dA, dB):
    """Block-cyclic shift: C|i,j> = |i+1, j+1> for i < dA-1 and C|dA-1, j> = |0, j+1-dA>."""
    N = dA * dB
    C = np.zeros((N, N))
    for i in range(dA):
        for j in range(dB):
            i2 = (i + 1) % dA
            j2 = (j + 1 - (dA if i == dA - 1 else 0)) % dB
            C[i2 * dB + j2, i * dB + j] = 1
    return C


def T_op(dA, dB, mu, nu):
    wB = np.exp(2j * np.pi / dB)
    XA, XB = shift(dA), shift(dB)
    Zt, ZB = clock(dB, dA, wB), clock(dB, dB, wB)
    mp = np.linalg.matrix_power
    return np.kron(mp(XA, mu) @ mp(Zt, nu), mp(XB, mu) @ mp(ZB.conj(), nu))


def Phi1(rho, dA, dB):
    out = np.zeros_like(rho, dtype=complex)
    for mu in range(dA):
        for nu in range(dB):
            T = T_op(dA, dB, mu, nu)
            out += T @ rho @ T.conj().T
    return out / (dA * dB)


def Pi_pinch(rho, B):
    p = np.real(np.einsum("ik,ij,jk->k", B.conj(), rho, B))
    return (B * p) @ B.conj().T


def Pi_twirl(rho, dA, dB):
    C, D = op_C(dA, dB), op_D(dA, dB)
    out = np.zeros_like(rho, dtype=complex)
    mp = np.linalg.matrix_power
    for mu in range(dA):
        for nu in range(dB):
            U = mp(C, mu) @ mp(D, nu)
            out += U @ rho @ U.conj().T
    return out / (dA * dB)


def PT_A(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(2, 1, 0, 3).reshape(dA * dB, dA * dB)


def negativity(rho, dA, dB):
    ev = np.linalg.eigvalsh(PT_A(rho, dA, dB))
    return float(-ev[ev < 0].sum())


def proj(v):
    v = np.asarray(v, dtype=complex).reshape(-1)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def rand_state(N, rng):
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    r = A @ A.conj().T
    return r / np.trace(r)


def realign(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)


def ccnr_value(rho, dA, dB):
    """MWKB normalisation: separable states obey CCNR <= sqrt(dA dB)."""
    return math.sqrt(dA * dB) * np.linalg.svd(realign(rho, dA, dB), compute_uv=False).sum()


DIMS_UNEQUAL = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6)]
DIMS_EQUAL = [(2, 2), (3, 3), (4, 4)]
rng = np.random.default_rng(20261117)

# ==============================================================================================
header("A1. Generalised Bell basis, the commuting pair (C, D) and Pi as an abelian twirl")
# ==============================================================================================
for dA, dB in DIMS_UNEQUAL + DIMS_EQUAL:
    N = dA * dB
    B = bell_basis(dA, dB)
    C, D = op_C(dA, dB), op_D(dA, dB)
    wA, wB = np.exp(2j * np.pi / dA), np.exp(2j * np.pi / dB)
    ok = np.allclose(B.conj().T @ B, np.eye(N), atol=TOL)
    ok &= np.allclose(C @ D, D @ C, atol=TOL)
    ok &= np.allclose(np.linalg.matrix_power(C, dA), np.eye(N)) and np.allclose(np.linalg.matrix_power(D, dB), np.eye(N))
    eC = np.array([wA ** (-a) for a in range(dA) for b in range(dB)])
    eD = np.array([wB ** (-b) for a in range(dA) for b in range(dB)])
    ok &= np.allclose(C @ B, B * eC, atol=TOL) and np.allclose(D @ B, B * eD, atol=TOL)
    rho = rand_state(N, rng)
    ok &= np.allclose(Pi_pinch(rho, B), Pi_twirl(rho, dA, dB), atol=TOL)
    # corrected Fourier representation: rho_P = (dA dB)^{-1} sum lambda_{mu nu} C^mu D^nu
    P = rng.random((dA, dB)); P /= P.sum()
    lam = np.array([[sum(P[a, b] * wA ** (a * mu) * wB ** (b * nu) for a in range(dA) for b in range(dB))
                     for nu in range(dB)] for mu in range(dA)])
    rhoP = (B * P.reshape(-1)) @ B.conj().T
    mp = np.linalg.matrix_power
    fourier_C = sum(lam[mu, nu] * mp(C, mu) @ mp(D, nu) for mu in range(dA) for nu in range(dB)) / N
    ok &= np.allclose(rhoP, fourier_C, atol=TOL)
    check(f"{dA}x{dB}: Bell basis orthonormal; [C,D]=0; C,D diagonal in Bell basis; Pi = <C,D>-twirl; "
          f"rho_P = sum lambda C^mu D^nu/(dA dB)", ok)
    # C is the local operator X_A (x) X_B iff dA = dB
    check(f"{dA}x{dB}: C == X_A(x)X_B is {dA == dB}", np.allclose(C, op_S(dA, dB)) == (dA == dB))

# ==============================================================================================
header("A2. MWKB Eqs. (10)/(17), (21) and Proposition 1 [Phi_1 = Pi] fail for dA < dB, hold for dA = dB")
# ==============================================================================================
for dA, dB in DIMS_UNEQUAL + DIMS_EQUAL:
    N = dA * dB
    B = bell_basis(dA, dB)
    wA, wB = np.exp(2j * np.pi / dA), np.exp(2j * np.pi / dB)
    eq = dA == dB
    # (21), second identity: Ztilde^nu X_A^mu = w_B^{mu nu} X_A^mu Ztilde^nu  (tested at mu = nu = 1)
    Zt, XA = clock(dB, dA, wB), shift(dA)
    id21 = np.allclose(Zt @ XA, wB * XA @ Zt, atol=TOL)
    # (17) with P = delta_{00}: (dA dB)^{-1} sum_{mu,nu} T(mu,nu) should be |phi00><phi00|
    rhs17 = sum(T_op(dA, dB, mu, nu) for mu in range(dA) for nu in range(dB)) / N
    phi00 = proj(B[:, 0])
    id17 = np.allclose(rhs17, phi00, atol=TOL)
    herm17 = np.allclose(rhs17, rhs17.conj().T, atol=TOL)
    # Proposition 1 at q = 1
    out = Phi1(phi00, dA, dB)
    prop1 = np.allclose(out, phi00, atol=TOL)
    fid = np.real(np.trace(phi00 @ out))
    fid_formula = (dA + 1) * (2 * dA + 1) / (6 * dA ** 2)
    rho = rand_state(N, rng)
    prop1_rand = np.allclose(Phi1(rho, dA, dB), Pi_pinch(rho, B), atol=TOL)
    idem = np.allclose(Phi1(Phi1(rho, dA, dB), dA, dB), Phi1(rho, dA, dB), atol=TOL)
    check(f"{dA}x{dB}: identity (21b) holds = {id21}; Eq.(17)[P=delta00] holds = {id17} (hermitian rhs = {herm17}); "
          f"Phi_1 = Pi holds = {prop1_rand}; Phi_1 idempotent = {idem}",
          (id21 == eq) and (id17 == eq) and (prop1 == eq) and (prop1_rand == eq) and (idem == eq))
    if not eq:
        check(f"{dA}x{dB}: <phi00|Phi_1(phi00)|phi00> = (dA+1)(2dA+1)/(6 dA^2) = {fid_formula:.6f}",
              abs(fid - fid_formula) < TOL, f"computed {fid:.6f}")
    # structure: Phi_1 = (1/dA) sum_{mu<dA} Ad_{S^mu} o P_D,  Pi = (1/dA) sum_{mu<dA} Ad_{C^mu} o P_D
    D, S, C = op_D(dA, dB), op_S(dA, dB), op_C(dA, dB)
    mp = np.linalg.matrix_power
    PD = sum(mp(D, nu) @ rho @ mp(D, nu).conj().T for nu in range(dB)) / dB
    blk = np.array([[(j - i) % dB for i in range(dA) for j in range(dB)]]).reshape(-1)
    mask = blk[:, None] == blk[None, :]
    okPD = np.allclose(PD, rho * mask, atol=TOL)
    phi_struct = sum(mp(S, mu) @ PD @ mp(S, mu).T for mu in range(dA)) / dA
    pi_struct = sum(mp(C, mu) @ PD @ mp(C, mu).T for mu in range(dA)) / dA
    check(f"{dA}x{dB}: P_D = block projection onto (+)_beta V_beta; Phi_1 = avg Ad_S^mu o P_D; Pi = avg Ad_C^mu o P_D",
          okPD and np.allclose(phi_struct, Phi1(rho, dA, dB), atol=TOL) and np.allclose(pi_struct, Pi_pinch(rho, B), atol=TOL))

# ==============================================================================================
header("A3. The group G = <T(mu,nu)> = <S, D>: order, commutant, fixed points and limit of Phi_1")
# ==============================================================================================


def group_order(dA, dB, cap=200000):
    """Closure of <S, D> as monomial matrices: element = (perm, phase exponents mod dB)."""
    N = dA * dB
    permS = tuple(((k // dB + 1) % dA) * dB + (k % dB + 1) % dB for k in range(N))
    gens = [(permS, tuple([0] * N)), (tuple(range(N)), tuple((k // dB - k % dB) % dB for k in range(N)))]

    def mul(g, h):  # (g h)|k> = w^{h.ph[k] + g.ph[h.perm[k]]} |g.perm[h.perm[k]]>
        return (tuple(g[0][h[0][k]] for k in range(N)), tuple((h[1][k] + g[1][h[0][k]]) % dB for k in range(N)))

    e = (tuple(range(N)), tuple([0] * N))
    seen, frontier = {e}, [e]
    while frontier:
        new = []
        for x in frontier:
            for g in gens:
                y = mul(g, x)
                if y not in seen:
                    seen.add(y)
                    new.append(y)
        frontier = new
        if len(seen) > cap:
            return None
    return len(seen)


print("   dA dB | gcd lcm | |G| brute force | lcm*dB*(dB/gcd)^dA | dim Fix(Phi_1) | 2nd largest |eig| of Phi_1")
for dA, dB in DIMS_UNEQUAL + DIMS_EQUAL:
    N = dA * dB
    g, l = math.gcd(dA, dB), math.lcm(dA, dB)
    formula = l * dB * (dB // g) ** dA
    order = group_order(dA, dB)
    # superoperator of Phi_1 acting on vec(rho) (row-major): sum T (x) conj(T)
    Ts = [T_op(dA, dB, mu, nu) for mu in range(dA) for nu in range(dB)]
    sup = sum(np.kron(T, T.conj()) for T in Ts) / N
    ev = np.linalg.eigvals(sup)
    n_one = int(np.sum(np.abs(ev - 1) < 1e-8))
    n_peripheral = int(np.sum(np.abs(ev) > 1 - 1e-8))
    second = float(np.sort(np.abs(ev))[-n_one - 1])
    # orbit twirl E(rho) = sum_r tr(Q_r rho) Q_r / tr(Q_r), Q_r = projector on {|i,j>: j - i = r mod gcd}
    E = np.zeros((N * N, N * N))
    cls = [(k % dB - k // dB) % g for k in range(N)]
    for r in range(g):
        q = np.array([1.0 if c == r else 0.0 for c in cls])
        vecQ = np.diag(q).reshape(-1)
        E += np.outer(vecQ, vecQ) / q.sum()
    okE = np.allclose(E @ sup, E, atol=TOL) and np.allclose(sup @ E, E, atol=TOL) and np.allclose(E @ E, E, atol=TOL)
    print(f"   {dA:2d} {dB:2d} | {g:3d} {l:3d} | {order!s:>15} | {formula:>18} | {n_one:>14} | {second:.6f}")
    check(f"{dA}x{dB}: |G| = lcm*dB*(dB/gcd)^dA = {formula}", order == formula)
    if dA < dB:
        check(f"{dA}x{dB}: dim Fix(Phi_1) = gcd = {g}; eigenvalue 1 is the only peripheral one; "
              f"orbit twirl E satisfies E Phi_1 = Phi_1 E = E = E^2 (so Phi_1^n -> E)",
              n_one == g and n_peripheral == g and okE and second < 1 - 1e-6)
    else:  # control: for dA = dB the channel is the Bell projection itself, Fix = span of Bell projectors
        check(f"{dA}x{dB} (control): dim Fix(Phi_1) = dA*dB = {N} and Phi_1 is a projection",
              n_one == N and np.allclose(sup @ sup, sup, atol=TOL))

# ==============================================================================================
header("B1. Exact 2x3 counterexample: Pi(|+>|+_01>) has partial-transpose eigenvalue -1/8")
# ==============================================================================================


def Pi_exact(rho, dA, dB):
    """Exact rational Pi = (1/dA) sum_mu Ad_{C^mu} o (block projection).  rho: N x N list of Fractions."""
    N = dA * dB
    blk = [(k % dB - k // dB) % dB for k in range(N)]
    cur = [[rho[r][c] if blk[r] == blk[c] else Fraction(0) for c in range(N)] for r in range(N)]
    permC = [((k // dB + 1) % dA) * dB + (k % dB + 1 - (dA if k // dB == dA - 1 else 0)) % dB for k in range(N)]
    out = [[Fraction(0)] * N for _ in range(N)]
    for _ in range(dA):
        for r in range(N):
            for c in range(N):
                out[r][c] += cur[r][c]
        nxt = [[Fraction(0)] * N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                nxt[permC[r]][permC[c]] = cur[r][c]
        cur = nxt
    return [[x / dA for x in row] for row in out]


def PT_exact(rho, dA, dB):
    N = dA * dB
    out = [[Fraction(0)] * N for _ in range(N)]
    for i in range(dA):
        for j in range(dB):
            for k in range(dA):
                for l in range(dB):
                    out[k * dB + j][i * dB + l] = rho[i * dB + j][k * dB + l]
    return out


def family_state_exact(dA, dB):
    """|a>|b> with a = (|0>+|1>)/sqrt2 in C^dA, b = (|0>+|1>)/sqrt2 in C^dB (density matrix, rational)."""
    N = dA * dB
    supp = [i * dB + j for i in (0, 1) for j in (0, 1)]
    rho = [[Fraction(0)] * N for _ in range(N)]
    for r in supp:
        for c in supp:
            rho[r][c] = Fraction(1, 4)
    return rho


dA, dB = 2, 3
sig = Pi_exact(family_state_exact(dA, dB), dA, dB)
Bf = bell_basis(dA, dB)
p = np.real(np.diag(Bf.conj().T @ np.array(sig, dtype=float) @ Bf)).reshape(dA, dB)
print("   Bell populations p_{alpha beta} of Pi(|+>|+_01>):")
print("   " + str(np.round(p, 12).tolist()))
check("2x3: populations are p00=1/2, p01=p02=p11=p12=1/8, p10=0",
      np.allclose(p, [[0.5, 0.125, 0.125], [0.0, 0.125, 0.125]], atol=TOL))
check("2x3: exact Pi agrees with the pinching in the Bell basis",
      np.allclose(np.array(sig, dtype=float), Pi_pinch(np.array(family_state_exact(2, 3), dtype=float), Bf), atol=TOL))
sigG = sp.Matrix(PT_exact(sig, dA, dB))
ev_exact = sigG.eigenvals()
print("   exact spectrum of Pi(rho)^{T_A}:", {str(k): v for k, v in ev_exact.items()})
check("2x3: exact spectrum of the partial transpose is {-1/8, 1/8 (x2), 1/4 (x2), 3/8}",
      ev_exact == {sp.Rational(-1, 8): 1, sp.Rational(1, 8): 2, sp.Rational(1, 4): 2, sp.Rational(3, 8): 1})
blk2 = sigG.extract([0 * 3 + 1, 1 * 3 + 0], [0 * 3 + 1, 1 * 3 + 0])
check("2x3: block of Pi(rho)^{T_A} on span{|01>,|10>} is [[1/8,1/4],[1/4,1/8]]",
      blk2 == sp.Matrix([[sp.Rational(1, 8), sp.Rational(1, 4)], [sp.Rational(1, 4), sp.Rational(1, 8)]]))

# ==============================================================================================
header("B2. Uniform family for all 2 <= dA < dB: lambda_min(Pi(rho)^{T_A}) = -1/(4 dA), negativity 1/(4 dA)")
# ==============================================================================================
all_ok, cnt = True, 0
for dA in range(2, 8):
    for dB in range(dA + 1, 10):
        N = dA * dB
        sig = Pi_exact(family_state_exact(dA, dB), dA, dB)
        G = PT_exact(sig, dA, dB)
        # exact eigenvector v = (|dA-1,0> - |0,dA-1>)/sqrt2 with eigenvalue -1/(4dA)
        v = [Fraction(0)] * N
        v[(dA - 1) * dB + 0] = Fraction(1)
        v[0 * dB + (dA - 1)] = Fraction(-1)
        Gv = [sum(G[r][c] * v[c] for c in range(N)) for r in range(N)]
        exact_eig = all(Gv[r] == Fraction(-1, 4 * dA) * v[r] for r in range(N))
        ev = np.linalg.eigvalsh(np.array(G, dtype=float))
        ok = exact_eig and abs(ev[0] + 1 / (4 * dA)) < TOL and ev[1] > -TOL and abs(-ev[ev < -TOL].sum() - 1 / (4 * dA)) < TOL
        all_ok &= ok
        cnt += 1
check(f"all {cnt} pairs 2 <= dA < dB <= 9 (dA <= 7): exact eigenvector equation, unique negative eigenvalue -1/(4dA)", all_ok)
# q-family: (1-q) rho + q Pi(rho) has <v|.^{T_A}|v> = -q/(4dA)
okq = True
for dA, dB in DIMS_UNEQUAL:
    N = dA * dB
    rho = np.array(family_state_exact(dA, dB), dtype=float)
    sig = np.array(Pi_exact(family_state_exact(dA, dB), dA, dB), dtype=float)
    v = np.zeros(N); v[(dA - 1) * dB] = 1 / np.sqrt(2); v[dA - 1] = -1 / np.sqrt(2)
    for q in (0.01, 0.3, 0.7, 1.0):
        val = v @ PT_A((1 - q) * rho + q * sig, dA, dB) @ v
        okq &= abs(val + q / (4 * dA)) < TOL
check("r.h.s. of MWKB (20) is entangling for every q in (0,1]: <v|[(1-q)rho + q Pi(rho)]^{T_A}|v> = -q/(4dA)", okq)
# controls: equal dimensions -> Pi is an LOCC twirl, so product states stay PPT
okc = True
for d in (2, 3, 4, 5):
    B = bell_basis(d, d)
    sig = np.array(Pi_exact(family_state_exact(d, d), d, d), dtype=float)
    okc &= np.linalg.eigvalsh(PT_A(sig, d, d))[0] > -TOL
    for _ in range(200):
        a = rng.normal(size=d) + 1j * rng.normal(size=d)
        b = rng.normal(size=d) + 1j * rng.normal(size=d)
        okc &= np.linalg.eigvalsh(PT_A(Pi_pinch(proj(np.kron(a, b)), B), d, d))[0] > -TOL
check("control dA = dB in {2,3,4,5}: Pi(product state) is PPT (family state and 200 random product states each)", okc)

# ==============================================================================================
header("B3. Maximal negativity of Pi(product state): numerical optimisation (best of many restarts)")
# ==============================================================================================


def max_negativity(dA, dB, restarts, seed):
    B = bell_basis(dA, dB)
    Bh = B.conj().T
    N = dA * dB

    def unpack(x):
        a = x[:dA] + 1j * x[dA:2 * dA]
        b = x[2 * dA:2 * dA + dB] + 1j * x[2 * dA + dB:]
        return a / np.linalg.norm(a), b / np.linalg.norm(b)

    def neg(x):
        a, b = unpack(x)
        p = np.abs(Bh @ np.kron(a, b)) ** 2
        sig = (B * p) @ Bh
        ev = np.linalg.eigvalsh(PT_A(sig, dA, dB))
        return ev[ev < 0].sum()  # = -negativity

    r = np.random.default_rng(seed)
    best, bestx = 0.0, None
    starts = []
    x0 = np.zeros(2 * (dA + dB)); x0[0] = x0[1] = 1; x0[2 * dA] = x0[2 * dA + 1] = 1
    starts.append(x0)
    x1 = np.zeros(2 * (dA + dB)); x1[:dA] = 1; x1[2 * dA:2 * dA + dB] = 1
    starts.append(x1)
    starts += [r.normal(size=2 * (dA + dB)) for _ in range(restarts)]
    for x in starts:
        res = minimize(neg, x + 1e-3 * r.normal(size=x.size), method="BFGS", options={"gtol": 1e-9, "maxiter": 400})
        res = minimize(neg, res.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 4000})
        if res.fun < best:
            best, bestx = res.fun, res.x
    a, b = unpack(bestx)
    return -best, a, b


NEG_TABLE = {}
print("   dA dB | 1/(4dA) (family) | max negativity found | optimiser |a_i|^2 ; |b_j|^2")
for (dA, dB), nres in zip(DIMS_UNEQUAL, (60, 60, 40, 40, 30, 30, 20, 20)):
    val, a, b = max_negativity(dA, dB, nres, seed=1000 * dA + dB)
    NEG_TABLE[(dA, dB)] = val
    # independent re-evaluation of the optimiser through the <C,D>-twirl code path
    sig = Pi_twirl(proj(np.kron(a, b)), dA, dB)
    val2 = negativity(sig, dA, dB)
    print(f"   {dA:2d} {dB:2d} | {1 / (4 * dA):.6f}         | {val:.6f}             | "
          f"{np.round(np.abs(a) ** 2, 3).tolist()} ; {np.round(np.abs(b) ** 2, 3).tolist()}")
    check(f"{dA}x{dB}: optimised negativity {val:.6f} >= 1/(4dA) and reproduced by independent code path",
          val >= 1 / (4 * dA) - 1e-9 and abs(val - val2) < 1e-8)

# ==============================================================================================
header("C1. MWKB Eqs. (22)-(24): the displayed state is Pi(phi_theta), not Phi_1(phi_theta)")
# ==============================================================================================
dA, dB = 2, 3
B = bell_basis(dA, dB)
ok_pi, ok_phi_differs, ok_npt = True, True, True
for th in np.linspace(0, 1, 11):
    v = np.zeros(6, dtype=complex); v[0] = 1; v[1 * 3 + 1] = np.exp(1j * th * np.pi)
    rho = proj(v)
    claimed = np.cos(th * np.pi / 2) ** 2 * proj(B[:, 0]) + np.sin(th * np.pi / 2) ** 2 * proj(B[:, 1 * 3 + 0])
    ok_pi &= np.allclose(Pi_pinch(rho, B), claimed, atol=TOL)
    out = Phi1(rho, dA, dB)
    ok_phi_differs &= not np.allclose(out, claimed, atol=1e-6)
    ok_npt &= abs(np.linalg.eigvalsh(PT_A(out, dA, dB))[0] + math.sqrt(2) / 4) < TOL
check("2x3: cos^2 phi00 + sin^2 phi10 equals Pi(phi_theta) for all theta", ok_pi)
check("2x3: it differs from Phi_1(phi_theta) [MWKB (19)] for all theta", ok_phi_differs)
check("2x3: the true Phi_1(phi_theta) has lambda_min(PT) = -sqrt(2)/4 for every theta (NPT also at theta = 1/2)", ok_npt)
half = np.zeros(6, dtype=complex); half[0] = 1; half[4] = 1j
check("2x3: Pi(phi_{1/2}) = (|00><00| + |11><11|)/2 is separable although phi_{1/2} is maximally entangled",
      np.allclose(Pi_pinch(proj(half), B), np.diag([0.5, 0, 0, 0, 0.5, 0]), atol=TOL))

# ==============================================================================================
header("C2. Downstream results of MWKB that do not use Proposition 1 (Sections III-V)")
# ==============================================================================================


def rho_from_P(P):
    P = np.asarray(P, dtype=float)
    dA, dB = P.shape
    B = bell_basis(dA, dB)
    return (B * (P / P.sum()).reshape(-1)) @ B.conj().T


P1 = [[1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1]]   # MWKB (38)
P2 = [[1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]   # MWKB (39)
for name, P in (("P(1)", P1), ("P(2)", P2)):
    r = rho_from_P(P)
    ptmin = np.linalg.eigvalsh(PT_A(r, 4, 6))[0]
    cc = ccnr_value(r, 4, 6)
    check(f"4x6 {name}: PPT (min PT eigenvalue {ptmin:.2e}) and CCNR = sqrt(24) + {cc - math.sqrt(24):.4f} > sqrt(24)",
          ptmin > -TOL and abs(cc - math.sqrt(24) - 0.554) < 5e-4)
P30 = [[1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0]]                           # MWKB (30)
r = rho_from_P(P30)
check("4x4 Eq.(30): PPT and CCNR = 6", np.linalg.eigvalsh(PT_A(r, 4, 4))[0] > -TOL and abs(ccnr_value(r, 4, 4) - 6) < 1e-9)


def dichotomous_search(dA, dB):
    B = bell_basis(dA, dB)
    N = dA * dB
    projs = [proj(B[:, k]) for k in range(N)]
    found = []
    for bits in range(1, 2 ** N):
        S = [k for k in range(N) if bits >> k & 1]
        r = sum(projs[k] for k in S) / len(S)
        if np.linalg.eigvalsh(PT_A(r, dA, dB))[0] > -1e-9 and ccnr_value(r, dA, dB) > math.sqrt(N) + 1e-7:
            found.append(frozenset((k // dB, k % dB) for k in S))
    classes, seen = 0, set()
    for S in found:
        if S in seen:
            continue
        classes += 1
        for s in range(dA):
            for t in range(dB):
                seen.add(frozenset(((a + s) % dA, (b + t) % dB) for a, b in S))
    return len(found), classes


for dA, dB, expect in ((2, 3, 0), (2, 4, 0), (2, 5, 0), (3, 4, 2)):
    n, c = dichotomous_search(dA, dB)
    check(f"{dA}x{dB}: exhaustive dichotomous search finds {n} PPT states detected by CCNR, "
          f"{c} classes up to cyclic row/column shifts (MWKB: {expect})", c == expect)


def hw_basis(d):
    X, Z = shift(d), clock(d)
    mp = np.linalg.matrix_power
    return [mp(X, a) @ mp(Z, b) for a in range(d) for b in range(d)]


def corr_matrix(rho, dA, dB):
    BA, BB = hw_basis(dA), hw_basis(dB)
    return np.array([[np.trace(np.kron(a, b).conj().T @ rho) for b in BB] for a in BA])


def ssc_g(Cm, dA, dB, x, y):
    M = Cm.copy()
    M[0, :] *= x
    M[:, 0] *= y
    return math.sqrt(dA - 1 + x * x) * math.sqrt(dB - 1 + y * y) - np.linalg.svd(M, compute_uv=False).sum()


rho1 = rho_from_P(P1)
C1 = corr_matrix(rho1, 4, 6)
C0 = corr_matrix(np.eye(24) / 24, 4, 6)
best_eps = 0.0
for x in np.linspace(0.05, 2, 40):
    for y in np.linspace(0.05, 2, 40):
        if ssc_g((1 - best_eps) * C1 + best_eps * C0, 4, 6, x, y) >= 0:
            continue
        lo, hi = best_eps, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if ssc_g((1 - mid) * C1 + mid * C0, 4, 6, x, y) < 0:
                lo = mid
            else:
                hi = mid
        best_eps = max(best_eps, lo)
Ce = (1 - 0.129) * C1 + 0.129 * C0
check(f"4x6 rho_1 [MWKB (38)]: SSC noise threshold on a 40x40 (x,y) grid = {best_eps:.4f} (MWKB: ~0.1295)",
      abs(best_eps - 0.1295) < 1.5e-3)
check(f"4x6 rho_1 at eps = 0.129: CCNR g(1,1) = {ssc_g(Ce, 4, 6, 1, 1):+.4f} > 0 and de Vicente g(0,0) = "
      f"{ssc_g(Ce, 4, 6, 0, 0):+.4f} > 0 (not detected)", ssc_g(Ce, 4, 6, 1, 1) > 0 and ssc_g(Ce, 4, 6, 0, 0) > 0)
# MWKB (67): moduli of the correlation matrix of |phi00> in 2x3
Cphi = np.abs(corr_matrix(proj(bell_basis(2, 3)[:, 0]), 2, 3))
vals = sorted(np.round(Cphi.reshape(-1), 9).tolist(), reverse=True)
expect = sorted([1.0] + [math.sqrt(3) / 2] * 2 + [0.5] * 14 + [0.0] * 19, reverse=True)
check("2x3 |phi00>: moduli of correlation-matrix entries are {1, sqrt3/2 (x2), 1/2 (x14), 0 (x19)} as in MWKB (67)",
      np.allclose(vals, np.round(expect, 9), atol=1e-8))

# ==============================================================================================
header("D. Local symmetries of the Bell basis: only the dephasing group {D^nu} survives when dA < dB")
# ==============================================================================================
for dA, dB in [(2, 3), (2, 4), (3, 4), (3, 5), (2, 2), (3, 3)]:
    N = dA * dB
    blocks = [frozenset((i, (i + b) % dB) for i in range(dA)) for b in range(dB)]
    good = []
    for pi in itertools.permutations(range(dA)):
        for tau in itertools.permutations(range(dB)):
            if all(frozenset((pi[i], tau[j]) for i, j in blk) == blk for blk in blocks):
                good.append((pi, tau))
    expect = 1 if dA < dB else dA
    check(f"{dA}x{dB}: permutation pairs (pi, tau) preserving every block V_beta: {len(good)} (expected {expect})",
          len(good) == expect)
    # D^nu fixes every Bell projector; S = X_A (x) X_B does so iff dA = dB
    B = bell_basis(dA, dB)
    D, S = op_D(dA, dB), op_S(dA, dB)
    fixD = all(np.allclose(D @ proj(B[:, k]) @ D.conj().T, proj(B[:, k]), atol=TOL) for k in range(N))
    fixS = all(np.allclose(S @ proj(B[:, k]) @ S.T, proj(B[:, k]), atol=TOL) for k in range(N))
    check(f"{dA}x{dB}: D fixes all Bell projectors = {fixD}; X_A(x)X_B fixes all Bell projectors = {fixS}",
          fixD and (fixS == (dA == dB)))
# the block-dephasing twirl P_D is LOCC: product states stay PPT; and Pi o P_D = Pi
okP = True
for dA, dB in DIMS_UNEQUAL:
    B = bell_basis(dA, dB)
    blk = np.array([(k % dB - k // dB) % dB for k in range(dA * dB)])
    mask = blk[:, None] == blk[None, :]
    for _ in range(50):
        a = rng.normal(size=dA) + 1j * rng.normal(size=dA)
        b = rng.normal(size=dB) + 1j * rng.normal(size=dB)
        rho = proj(np.kron(a, b))
        okP &= np.linalg.eigvalsh(PT_A(rho * mask, dA, dB))[0] > -TOL
        okP &= np.allclose(Pi_pinch(rho * mask, B), Pi_pinch(rho, B), atol=TOL)
check("block dephasing P_D: P_D(product) is PPT (50 random product states per dimension pair) and Pi o P_D = Pi", okP)

# ----------------------------------------------------------------------------------------------
header("SUMMARY")
n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"  {len(RESULTS) - n_fail} / {len(RESULTS)} checks passed;  runtime {time.time() - T0:.1f} s")
print("  Table (max negativity of Pi(product state), numerical):")
for (dA, dB), v in NEG_TABLE.items():
    print(f"     {dA} x {dB}:  N_max >= {v:.6f}    family value 1/(4dA) = {1 / (4 * dA):.6f}")
if n_fail:
    print("  FAILED CHECKS:")
    for name, ok in RESULTS:
        if not ok:
            print("    -", name)
sys.exit(1 if n_fail else 0)
