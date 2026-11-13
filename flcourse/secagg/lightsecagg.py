"""LightSecAgg (So et al., MLSys 2022): secure aggregation with one-shot mask recovery.

Instead of secret-sharing *seeds* (SecAgg), every client secret-shares its *mask* with a
T-private MDS code. After dropouts, every survivor sends the aggregate of secret shares for the set
of all surviving clients. The server then decodes the aggregate mask (of the sum of all masks of all clients) directly
The server never has to regenerate per-client masks of dropped clients and only needs a single reconstruction (the decoding ^^)

Some parameters 

* ``N`` clients in the round 
* ``T`` privacy threshold: any ``T`` clients colluding with the server learn nothing about
  another client's input (information-theoretically).
* ``U`` target number of survivors, ``T < U <= N``; tolerates ``D = N - U`` dropouts.

Protocol (all arithmetic in the prime field F_p, p = 2^31 - 1, done by the ``galois`` library)::

    offline   client i draws a mask z_i in F_p^d, splits it into U-T pieces [z_i]_k and adds T
              random noise pieces [n_i]_k. With the U x N encoding matrix W (eq. 5 of the paper)
              client i sends  [z_i]_j = ([z_i]_1, ..., [z_i]_{U-T}, [n_i]_{U-T+1}, ..., [n_i]_U) . W_j
              (length d/(U-T)) to client j (encrypted, routed via the server).
    masking   client i sends  y_i = x_i + z_i  (mod p) (note that model x_i is quantised with some quantifier Q(x))
    recovery  server announces the survivors S; each surviving client j sends sum_{i in S} [z~_i]_j.
              Any U of them are (summed pieces) . W_cols with an invertible U x U matrix W_cols,
              so the server solves for the summed pieces; the first U-T give sum_{i in S} z_i.
    result    sum_{i in S} y_i - sum_{i in S} z_i = sum_{i in S} x_i  (mod p)

The encoding matrix is galois' Vandermonde matrix ``W[k, j] = a^(k*j)`` of a primitive
element ``a`` (the generator matrix of a Reed-Solomon code; the paper uses a Vandermonde
matrix as well, Appendix F). It is

* MDS: every U x U submatrix is a Vandermonde matrix in the distinct nodes a^j, hence
  invertible -> any U survivors suffice (dropout resiliency);
* T-private for every T < U: every T x T submatrix of its last T rows equals
  Vandermonde(a^j) . diag(a^(j (U-T))), hence invertible -> the T noise pieces hide the mask
  pieces from any T clients (privacy).

``tests/test_secagg.py`` checks both properties exhaustively for small N.

Quantisation Q(x) re-uses Flower's ``quantize``/``dequantize`` (``flcourse.secagg.flwr``), so
inputs are clipped to ``[-clipping_range, clipping_range]`` exactly as in SecAgg.

Note that i made a little scheme overview in the tutorial slides for LightSecAgg.
"""
from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import galois
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..bench import measure, timer
from .flwr.asymmetric import bytes_to_private_key, bytes_to_public_key, generate_key_pairs, \
    private_key_to_bytes, public_key_to_bytes
from .flwr.quantization import dequantize, quantize
from .flwr.symmetric_encryption import generate_shared_key

PRIME = 2**31 - 1          # Mersenne prime: field elements fit into uint32


# ----------------------------------------------------------------------------------
# finite field (galois)
# ----------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def field(p: int = PRIME):
    """The prime field GF(p) as a galois ``FieldArray`` class (exact modular arithmetic).
    Built once per process; the dummy computation compiles galois' numba kernels here, so
    that this one-time cost (~1 s) does not show up in the timed protocol stages."""
    GF = galois.GF(p)
    W = GF.Vandermonde(GF.primitive_element, 2, 3)
    np.linalg.inv(W[:, :2].T) @ (W.T @ GF.Ones((2, 4)))[:2]
    return GF


def t_private_mds_matrix(U: int, N: int, p: int = PRIME):
    """The U x N encoding matrix W of eq. (5): ``W[k, j] = a^(k*j)`` for a primitive element
    ``a`` of GF(p) -- MDS and T-private for every T < U (see the module docstring)."""
    GF = field(p)
    return GF.Vandermonde(GF.primitive_element, U, N)


@dataclass
class LightSecAggParams:
    privacy_threshold: float = 0.5     # T: int, or fraction of the clients in the round
    dropout_tolerance: float = 0.2     # D: int, or fraction of the clients in the round (U = N - D)
    clipping_range: float = 8.0
    quantization_range: int = 2**22
    prime: int = PRIME

    def round_parameters(self, n: int) -> Tuple[int, int]:
        """``(T, U)`` for a round with ``n`` clients (with all consistency checks)."""
        T = int(round(self.privacy_threshold * n)) if isinstance(self.privacy_threshold, float) \
            else int(self.privacy_threshold)
        D = int(math.floor(self.dropout_tolerance * n)) if isinstance(self.dropout_tolerance, float) \
            else int(self.dropout_tolerance)
        U = n - D
        if n < 2:
            raise ValueError("LightSecAgg needs at least 2 clients.")
        if not 0 <= T < U <= n:
            raise ValueError(f"LightSecAgg needs 0 <= T < U <= N; got T={T}, U={U} (N={n}, D={D}). "
                             f"Lower privacy_threshold or dropout_tolerance.")
        if self.prime > 2**31:
            raise ValueError("prime must be < 2^31 (field elements are sent as uint32)")
        if n * self.quantization_range >= self.prime:
            raise ValueError(f"Overflow: {n} clients x quantization_range {self.quantization_range} >= prime "
                             f"{self.prime}; the sum would wrap around F_p. Lower quantization_range.")
        if n >= self.prime - 1:
            raise ValueError("Too many clients for this prime (the nodes a^j of W must be distinct).")
        return T, U

    @property
    def bits_per_entry(self) -> int:
        return 32          # field elements are sent as uint32


def _aes_key(shared_key_b64: bytes) -> bytes:
    import base64
    return base64.urlsafe_b64decode(shared_key_b64)


# ----------------------------------------------------------------------------------
# client
# ----------------------------------------------------------------------------------
class LightSecAggClient:
    def __init__(self, nid: int, index: int, T: int, U: int, N: int, d: int, params: LightSecAggParams, W):
        self.nid, self.index = nid, index            # index j in 0..N-1 -> column W_j of the encoding matrix
        self.T, self.U, self.N, self.d = T, U, N, d
        self.params, self.p = params, params.prime
        self.W = W                                   # U x N T-private MDS matrix (galois array)
        self.piece_len = math.ceil(d / (U - T))
        self.received: Dict[int, np.ndarray] = {}   # sender nid -> coded mask piece [z~_sender]_me

    # stage 0: key agreement for the pairwise secure channels
    def advertise_key(self) -> bytes:
        sk, pk = generate_key_pairs()
        self.sk, self.pk = private_key_to_bytes(sk), public_key_to_bytes(pk)
        return self.pk

    def _channel_key(self, other_pk: bytes) -> bytes:
        return _aes_key(generate_shared_key(bytes_to_private_key(self.sk), bytes_to_public_key(other_pk)))

    # stage 1 (offline): draw mask, encode, send coded pieces
    def encode_mask(self, public_keys: Dict[int, bytes], index_of: Dict[int, int]) -> Dict[int, bytes]:
        
        # forming z_i
        p, k_data = self.p, self.U - self.T
        rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))
        self.mask = rng.integers(0, p, size=self.d, dtype=np.int64)
        # padding z_i
        padded = np.zeros(k_data * self.piece_len, dtype=np.int64)
        padded[:self.d] = self.mask
        
        # Here we split z_i into [z_i]_1 , ... [z_i]_{U-T}
        # Note that the size of each [z_i]_k is then the size of the resulting share
        # This sharing scheme is basically McElise-Sarwate (see lecture) with a specific scaling of the
        # payload pieces and additional randomness. This scaling it the MDS matrix W mentioned in the paper.
        # We then also put everything into one tuple such that ([z_i]_1, [z_i]_2, ... [z_i]_{U-T} , [n_i]_{U-T+1},...[n_i]_U)
        # where [n_i]_{U-T+1},...[n_i]_U are also randomly generated
        pieces = np.concatenate([padded.reshape(k_data, self.piece_len),
                                 rng.integers(0, p, size=(self.T, self.piece_len), dtype=np.int64)])
        
        GF = type(self.W)
        coded = (self.W.T @ GF(pieces)).view(np.ndarray).astype(np.int64)  # (N, piece_len): row j = pieces . W_j
        
        out = {}
        for nid, pk in public_keys.items():
            j = index_of[nid]
            if nid == self.nid:
                self.received[self.nid] = coded[j]
                continue
            aead = AESGCM(self._channel_key(pk))
            nonce = os.urandom(12)
            out[nid] = nonce + aead.encrypt(nonce, coded[j].astype(np.uint32).tobytes(), None)
        return out

    def receive_coded_pieces(self, ciphertexts: Dict[int, bytes], public_keys: Dict[int, bytes]) -> None:
        for src, ct in ciphertexts.items():
            aead = AESGCM(self._channel_key(public_keys[src]))
            plain = aead.decrypt(ct[:12], ct[12:], None)
            self.received[src] = np.frombuffer(plain, dtype=np.uint32).astype(np.int64)

    # stage 2: masked input
    def masked_input(self, vector: np.ndarray) -> np.ndarray:
        """
        Adds Q(x) to the mask z_i
        """
        q = quantize([np.asarray(vector, dtype=np.float64)], self.params.clipping_range,
                     self.params.quantization_range)[0].astype(np.int64)
        return ((q + self.mask) % self.p).astype(np.uint32)

    # stage 3: one-shot recovery -- sum of the coded pieces of all survivors
    def aggregate_coded_pieces(self, survivors: Sequence[int]) -> np.ndarray:
        missing = [s for s in survivors if s not in self.received]
        if missing:
            raise ValueError(f"client {self.nid} holds no coded piece of clients {missing}.")
        total = np.zeros(self.piece_len, dtype=np.int64)
        for s in survivors:
            total = (total + self.received[s]) % self.p
        return total.astype(np.uint32)


# ----------------------------------------------------------------------------------
# server
# ----------------------------------------------------------------------------------
class LightSecAggServer:
    def __init__(self, T: int, U: int, node_ids: Sequence[int], d: int, params: LightSecAggParams, W):
        self.T, self.U, self.d, self.params, self.p = T, U, d, params, params.prime
        self.node_ids = list(node_ids)
        self.index_of = {nid: j for j, nid in enumerate(self.node_ids)}
        self.W = W                                   # U x N T-private MDS matrix (galois array)
        self.public_keys: Dict[int, bytes] = {}
        self.mailbox: Dict[int, Dict[int, bytes]] = {nid: {} for nid in node_ids}
        self.masked_sum = np.zeros(d, dtype=np.int64)
        self.survivors: List[int] = []
        self.responses: Dict[int, np.ndarray] = {}

    def receive_masked_input(self, nid: int, y: np.ndarray) -> None:
        self.masked_sum = (self.masked_sum + y.astype(np.int64)) % self.p
        self.survivors.append(nid)

    def check_survivors(self) -> None:
        if len(self.survivors) < self.U:
            raise RuntimeError(f"LightSecAgg: only {len(self.survivors)} clients sent a masked input, "
                               f"need U={self.U}.")

    def decode(self) -> np.ndarray:
        """Interpolate the aggregate mask from U coded responses and return the sum of inputs."""
        if len(self.responses) < self.U:
            raise RuntimeError(f"LightSecAgg: only {len(self.responses)} recovery messages, need U={self.U}.")
        # putting the incoming responses into the correct order such that they match the interpolation position.
        # Note that we in essence simply interpolate here. For that input/output must align.
        senders = sorted(self.responses)[:self.U]
        cols = [self.index_of[s] for s in senders]

        # Look into Theorem 1 of the LightSecAgg paper (in particular the proof of it and the dropout-resilience guarantee)
        GF = type(self.W)
        V = GF(np.stack([self.responses[s].astype(np.int64) for s in senders]))     # (U, piece_len)
        pieces = np.linalg.inv(self.W[:, cols].T) @ V            # solve V = W_cols^T . pieces (MDS decoding)
        agg_mask = pieces[:self.U - self.T].view(np.ndarray).astype(np.int64).reshape(-1)[:self.d]
        return (self.masked_sum - agg_mask) % self.p

    def dequantize_average(self, total_q: np.ndarray) -> np.ndarray:
        """Implementing the inverse of Q(x)"""
        n = len(self.survivors)
        c, T = self.params.clipping_range, self.params.quantization_range
        total = dequantize([total_q.astype(np.int64)], c, T)[0]     # = sum x + n c - c
        return (total - (n - 1) * c) / n


def lightsecagg_aggregate(inputs: Dict[int, np.ndarray], params: LightSecAggParams,
                          dropouts: Optional[Dict[int, str]] = None, label: str = "lightsecagg"
                          ) -> Tuple[np.ndarray, dict]:
    """Securely average ``inputs``; ``dropouts``: ``{client: "masked_input" | "recovery"}``."""
    dropouts = dict(dropouts or {})
    node_ids = sorted(int(n) for n in inputs)
    N, d = len(node_ids), int(np.asarray(next(iter(inputs.values()))).size)
    T, U = params.round_parameters(N)
    field(params.prime)                            # one-time galois set-up, not part of the protocol
    with timer(f"{label}/0 setup [server]"):
        W = t_private_mds_matrix(U, N, params.prime)                     # (U, N), public
        server = LightSecAggServer(T, U, node_ids, d, params, W)
    clients = {nid: LightSecAggClient(nid, j, T, U, N, d, params, W) for j, nid in enumerate(node_ids)}
    measure({"T": T, "U": U, "N": N}, f"{label}/0 setup: config", src="server", dst=None)

    # ---- stage 0: key agreement for the pairwise channels -------------------------------
    for nid in node_ids:
        with timer(f"{label}/0 keys [client]", who=nid):
            pk = clients[nid].advertise_key()
        measure(pk, f"{label}/0 keys: public key", src=nid, dst="server")
        server.public_keys[nid] = pk
    for nid in node_ids:
        measure(server.public_keys, f"{label}/0 keys: public keys", src="server", dst=nid)

    # ---- stage 1 (offline): encode masks, route coded pieces ------------------------------
    for nid in node_ids:
        with timer(f"{label}/1 encode masks [client]", who=nid):
            cts = clients[nid].encode_mask(server.public_keys, server.index_of)
        measure(cts, f"{label}/1 coded mask pieces (up)", src=nid, dst="server")
        for dst, ct in cts.items():
            server.mailbox[dst][nid] = ct
    for nid in node_ids:
        measure(server.mailbox[nid], f"{label}/1 coded mask pieces (down)", src="server", dst=nid)
        with timer(f"{label}/1 decrypt pieces [client]", who=nid):
            clients[nid].receive_coded_pieces(server.mailbox[nid], server.public_keys)

    # ---- stage 2: masked inputs --------------------------------------------------------
    for nid in node_ids:
        if dropouts.get(nid) == "masked_input":
            continue
        with timer(f"{label}/2 masked input [client]", who=nid):
            y = clients[nid].masked_input(inputs[nid])
        measure(y, f"{label}/2 masked input", src=nid, dst="server", bits_per_entry=params.bits_per_entry)
        with timer(f"{label}/2 masked input [server]"):
            server.receive_masked_input(nid, y)
    server.check_survivors()

    # ---- stage 3: one-shot aggregate-mask recovery ---------------------------------------
    survivors = sorted(server.survivors)
    for nid in survivors:
        measure(survivors, f"{label}/3 recovery: survivor list", src="server", dst=nid)
        if dropouts.get(nid) == "recovery":
            continue
        with timer(f"{label}/3 recovery [client]", who=nid):
            r = clients[nid].aggregate_coded_pieces(survivors)
        measure(r, f"{label}/3 recovery: aggregated coded mask", src=nid, dst="server",
                bits_per_entry=params.bits_per_entry)
        server.responses[nid] = r
    with timer(f"{label}/3 decode [server]"):
        total_q = server.decode()
        average = server.dequantize_average(total_q)
    info = {"T": T, "U": U, "N": N, "survivors": len(survivors), "responses": len(server.responses),
            "aggregated_clients": survivors, "piece_len": clients[node_ids[0]].piece_len}
    return average, info


class LightSecAgg:
    """Convenience wrapper with the same interface as :class:`SecAggPlus`."""

    name = "lightsecagg"

    def __init__(self, privacy_threshold: float = 0.5, dropout_tolerance: float = 0.2, clipping_range: float = 8.0,
                 quantization_range: int = 2**22, prime: int = PRIME, seed: Optional[int] = None):
        self.params = LightSecAggParams(privacy_threshold, dropout_tolerance, clipping_range,
                                        int(quantization_range), int(prime))
        self.last_info: dict = {}

    def aggregate(self, inputs: Dict[int, np.ndarray], weights=None, dropouts: Optional[Dict[int, str]] = None
                  ) -> np.ndarray:
        if weights is not None:
            raise NotImplementedError("LightSecAgg here computes the unweighted mean (weights=None).")
        dropouts = {k: ("recovery" if v == "unmask" else v) for k, v in (dropouts or {}).items()}
        average, self.last_info = lightsecagg_aggregate(inputs, self.params, dropouts, label=self.name)
        return average

    def __repr__(self) -> str:
        return f"LightSecAgg({self.params})"
