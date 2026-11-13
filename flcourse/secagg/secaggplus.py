"""SecAgg (Bonawitz et al., CCS 2017) and SecAgg+ (Bell et al., CCS 2020).

A *single-process, stage-by-stage* simulation of Flower's ``SecAggPlusWorkflow``
(server) and ``secaggplus_mod`` (client). Also the consistency checks are copied.
We simplify the messaging (we still keep track of what is communicated and the computation time)


Protocol (``N(i)`` = neighbours of client ``i``; SecAgg: everybody; SecAgg+: a random
ring-shaped graph of ``num_shares`` clients)::

    stage 0  setup        server -> i : parameters          i -> server : pk1_i, pk2_i
    stage 1  share keys   server -> i : {pk_j : j in N(i)}  i -> server : Enc_{k_ij}(b_i[j], sk1_i[j])
                          b_i[j], sk1_i[j] = Shamir shares (threshold t) of the self-mask seed b_i
                          and of the mask key sk1_i; k_ij = HKDF(ECDH(sk2_i, pk2_j))
    stage 2  masked input server -> i : ciphertexts for i i -> server : y_i
                          y_i = Q(x_i) + PRG(b_i) + sum_{j in N(i), j<i} PRG(s_ij)
                                                   - sum_{j in N(i), j>i} PRG(s_ij)   (mod M)
                          s_ij = HKDF(ECDH(sk1_i, pk1_j)) = s_ji, so pairwise masks cancel
    stage 3  unmask       server -> i : who survived / dropped
                          i -> server : b_j[i] for survivors j, sk1_j[i] for dropped j
             server: reconstruct b_j (t shares) and remove PRG(b_j) of every survivor;
                     reconstruct sk1_j of dropped clients and remove their pairwise masks.

Flower implements SecAgg as SecAgg+ with ``num_shares = 1.0`` (= all clients); we do the
same (:class:`SecAgg`).

Note that masks of a client that dropped *after* sharing keys are removed only with 
neighbours that sent a masked input, and clients that dropped *before* sharing keys 
are ignored altogether.  (See (Bonawitz et al., CCS 2017) paper of Bonawitz sets: U1 > U2 > U3 > U4).
"""
from __future__ import annotations

import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..bench import measure, timer
from .flwr.asymmetric import (bytes_to_private_key, bytes_to_public_key, generate_key_pairs,
                              private_key_to_bytes, public_key_to_bytes)
from .flwr.ndarrays_arithmetic import (factor_combine, factor_extract, get_parameters_shape,
                                       parameters_addition, parameters_mod, parameters_multiply,
                                       parameters_subtraction)
from .flwr.quantization import dequantize, quantize
from .flwr.secaggplus_utils import (pseudo_rand_gen, share_keys_plaintext_concat,
                                    share_keys_plaintext_separate)
from .flwr.shamir import combine_shares, create_shares
from .flwr.symmetric_encryption import decrypt, encrypt, generate_shared_key

STAGES = ("setup", "share_keys", "masked_input", "unmask")


# This rather large parameter wrapper merely checks that all parameters make sense. This is from Flower library
@dataclass
class SecAggPlusParams:
    """Protocol parameters, from flower's ``SecAggPlusWorkflow``.

    num_shares:      int = number of neighbours incl. oneself (should be odd), or a float
                     fraction of the sampled clients. 1.0 = everybody = plain SecAgg.
    reconstruction_threshold: int, or float fraction of ``num_shares`` (Shamir threshold t).
    max_weight:      largest client weight (FedAvg: #samples). Large values cost precision!
    clipping_range:  inputs are clipped to [-c, c] before quantisation.
    quantization_range: inputs are quantised to integers in [0, T].
    modulus_range:   masks and sums live in Z_M (power of 2, M > T).
    """

    num_shares: float = 1.0
    reconstruction_threshold: float = 0.5
    max_weight: float = 1.0
    clipping_range: float = 8.0
    quantization_range: int = 4194304          # 2**22
    modulus_range: int = 4294967296            # 2**32

    def __post_init__(self):
        self._check_init_params()

    def _check_init_params(self) -> None:  # taken from Flower
        # Check `num_shares`
        if not isinstance(self.num_shares, (int, float)):
            raise TypeError("`num_shares` must be of type int or float.")
        if isinstance(self.num_shares, int):
            if self.num_shares == 1:
                self.num_shares = 1.0
            elif self.num_shares <= 2:
                raise ValueError("`num_shares` as an integer must be greater than 2.")
            elif self.num_shares > self.modulus_range / self.quantization_range:
                warnings.warn("A `num_shares` larger than `modulus_range / quantization_range` will "
                              "potentially cause overflow when computing the aggregated model parameters.")
        elif self.num_shares <= 0:
            raise ValueError("`num_shares` as a float must be greater than 0.")
        # Check `reconstruction_threshold`
        if not isinstance(self.reconstruction_threshold, (int, float)):
            raise TypeError("`reconstruction_threshold` must be of type int or float.")
        if isinstance(self.reconstruction_threshold, int):
            if self.reconstruction_threshold == 1:
                self.reconstruction_threshold = 1.0
            elif isinstance(self.num_shares, int) and self.reconstruction_threshold >= self.num_shares:
                raise ValueError("`reconstruction_threshold` must be less than `num_shares`.")
        elif not 0 < self.reconstruction_threshold <= 1:
            raise ValueError("If `reconstruction_threshold` is a float, it must be greater than 0 "
                             "and less than or equal to 1.")
        # Check `max_weight`
        if self.max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")
        # Check `quantization_range`
        if not isinstance(self.quantization_range, int) or self.quantization_range <= 0:
            raise ValueError("`quantization_range` must be an integer and greater than 0.")
        # Check `modulus_range`
        if not isinstance(self.modulus_range, int) or self.modulus_range <= self.quantization_range:
            raise ValueError("`modulus_range` must be an integer and greater than `quantization_range`.")
        if bin(self.modulus_range).count("1") != 1:
            raise ValueError("`modulus_range` must be a power of 2.")
        if self.modulus_range > 2**62:
            raise ValueError("`modulus_range` must be at most 2**62 in this simulation.")

    @property
    def bits_per_entry(self) -> int:
        """Bits needed to send one masked entry (an element of Z_M)."""
        return int(math.log2(self.modulus_range))


def round_parameters(params: SecAggPlusParams, num_samples: int) -> Tuple[int, int]:
    """``(num_shares, threshold)`` for this round, exactly as Flower's ``setup_stage``."""
    if num_samples < 2:
        raise ValueError("The number of samples should be greater than 1.")
    if isinstance(params.num_shares, float):
        num_shares = round(params.num_shares * num_samples)
        if num_shares < num_samples and num_shares & 1 == 0:      # if even
            num_shares += 1
        if num_shares <= 2:                                          # if too small
            num_shares = num_samples
    else:
        num_shares = params.num_shares
    if isinstance(params.reconstruction_threshold, float):
        threshold = round(params.reconstruction_threshold * num_shares)
        threshold = max(threshold, 2)                                # avoid too small threshold
    else:
        threshold = params.reconstruction_threshold
    if num_samples != num_shares and num_shares & 1 == 0:
        warnings.warn("Number of shares in the SecAgg+ protocol should be odd.")
        num_shares += 1
    # flcourse: refuse parameters that would wrap around Z_M (Flower only warns via num_shares)
    if num_samples * params.quantization_range >= params.modulus_range:
        raise ValueError(f"Overflow: {num_samples} clients x quantization_range {params.quantization_range} "
                         f">= modulus_range {params.modulus_range}. The sum would wrap around Z_M; "
                         f"increase modulus_range or lower quantization_range.")
    return num_shares, threshold


def build_neighbours(node_ids: Sequence[int], num_shares: int, rng: random.Random) -> Dict[int, Set[int]]:
    """Flower's SecAgg+ graph: shuffle the ids, connect everybody to the ``num_shares // 2``
    clients on each side of a ring (itself included). ``num_shares >= n`` -> complete graph."""
    ids = list(node_ids)
    rng.shuffle(ids)
    n, half = len(ids), num_shares >> 1
    return {nid: {ids[(idx + off) % n] for off in range(-half, half + 1)} for idx, nid in enumerate(ids)}


# ----------------------------------------------------------------------------------
# client side  (Flower: flwr/client/mod/secure_aggregation/secaggplus_mod.py)
# ----------------------------------------------------------------------------------
class SecAggPlusClient:
    """The four client stages of ``secaggplus_mod`` as methods (one object per round)."""

    def __init__(self, nid: int):
        self.nid = int(nid)

    # stage 0 ----------------------------------------------------------------------------
    def setup(self, config: dict) -> Tuple[bytes, bytes]:
        self.sample_num = int(config["sample_num"])
        self.share_num = int(config["share_num"])
        self.threshold = int(config["threshold"])
        self.clipping_range = float(config["clipping_range"])
        self.target_range = int(config["target_range"])
        self.mod_range = int(config["mod_range"])
        self.max_weight = float(config["max_weight"])
        # node id -> shares of *its* secrets that this client holds
        self.rd_seed_share_dict: Dict[int, bytes] = {}
        self.sk1_share_dict: Dict[int, bytes] = {}
        # node id -> shared secret (from sk2) used to encrypt shares
        self.ss2_dict: Dict[int, bytes] = {}
        # two key pairs: sk1/pk1 for pairwise masks, sk2/pk2 for encrypting shares
        sk1, pk1 = generate_key_pairs()
        sk2, pk2 = generate_key_pairs()
        self.sk1, self.pk1 = private_key_to_bytes(sk1), public_key_to_bytes(pk1)
        self.sk2, self.pk2 = private_key_to_bytes(sk2), public_key_to_bytes(pk2)
        return self.pk1, self.pk2

    # stage 1 ----------------------------------------------------------------------------
    def share_keys(self, public_keys: Dict[int, Tuple[bytes, bytes]]) -> Dict[int, bytes]:
        """Returns ``{destination node id: ciphertext}``."""
        self.public_keys_dict = dict(public_keys)
        # Check if the size is larger than threshold
        if len(self.public_keys_dict) < self.threshold:
            raise ValueError("Available neighbours number smaller than threshold")
        # Check if all public keys are unique
        pk_list: List[bytes] = []
        for pk1, pk2 in self.public_keys_dict.values():
            pk_list += [pk1, pk2]
        if len(set(pk_list)) != len(pk_list):
            raise ValueError("Some public keys are identical")
        # Check if public keys of this client are correct in the dictionary
        if self.public_keys_dict[self.nid][0] != self.pk1 or self.public_keys_dict[self.nid][1] != self.pk2:
            raise ValueError("Own public keys are displayed in dict incorrectly, should not happen!")

        # Generate the private mask seed
        self.rd_seed = os.urandom(32)
        # Create shares for the private mask seed and the first private key
        b_shares = create_shares(self.rd_seed, self.threshold, self.share_num)
        sk1_shares = create_shares(self.sk1, self.threshold, self.share_num)

        ciphertexts: Dict[int, bytes] = {}
        sk2 = bytes_to_private_key(self.sk2)
        for idx, (nid, (_, pk2)) in enumerate(self.public_keys_dict.items()):
            if nid == self.nid:
                self.rd_seed_share_dict[self.nid] = b_shares[idx]
                self.sk1_share_dict[self.nid] = sk1_shares[idx]
            else:
                shared_key = generate_shared_key(sk2, bytes_to_public_key(pk2))
                self.ss2_dict[nid] = shared_key
                plaintext = share_keys_plaintext_concat(self.nid, nid, b_shares[idx], sk1_shares[idx])
                ciphertexts[nid] = encrypt(shared_key, plaintext)
        return ciphertexts

    # stage 2 ----------------------------------------------------------------------------
    def collect_masked_vectors(self, ciphertexts: Dict[int, bytes], parameters: List[np.ndarray],
                               num_examples: float = 1.0) -> List[np.ndarray]:
        """Decrypt the received shares, then quantise, weight and mask ``parameters``."""
        available_clients: List[int] = []
        if len(ciphertexts) + 1 < self.threshold:
            raise ValueError("Not enough available neighbour clients.")
        # Decrypt ciphertexts, verify their sources, and store shares.
        for src, ciphertext in ciphertexts.items():
            plaintext = decrypt(self.ss2_dict[src], ciphertext)
            actual_src, dst, rd_seed_share, sk1_share = share_keys_plaintext_separate(plaintext)
            available_clients.append(src)
            if src != actual_src:
                raise ValueError(f"Node {self.nid}: received ciphertext from {actual_src} instead of {src}.")
            if dst != self.nid:
                raise ValueError(f"Node {self.nid}: received an encrypted message for Node {dst} from Node {src}.")
            self.rd_seed_share_dict[src] = rd_seed_share
            self.sk1_share_dict[src] = sk1_share

        # Weighting factor (FedAvg weight = num_examples)
        ratio = num_examples / self.max_weight
        if ratio > 1:
            warnings.warn(f"Potential overflow warning: the provided weight ({num_examples}) exceeds the "
                          f"specified max_weight ({self.max_weight}). This may lead to overflow issues.")
        q_ratio = round(ratio * self.target_range)
        dq_ratio = q_ratio / self.target_range
        parameters = parameters_multiply(parameters, dq_ratio)

        # Quantize parameter update (vector)
        quantized = quantize(parameters, self.clipping_range, self.target_range)
        quantized = factor_combine(q_ratio, quantized)
        dimensions_list = [a.shape for a in quantized]

        # Add private mask
        private_mask = pseudo_rand_gen(self.rd_seed, self.mod_range, dimensions_list)
        quantized = parameters_addition(quantized, private_mask)

        sk1 = bytes_to_private_key(self.sk1)
        for node_id in available_clients:
            # Add pairwise masks
            shared_key = generate_shared_key(sk1, bytes_to_public_key(self.public_keys_dict[node_id][0]))
            pairwise_mask = pseudo_rand_gen(shared_key, self.mod_range, dimensions_list)
            if self.nid > node_id:
                quantized = parameters_addition(quantized, pairwise_mask)
            else:
                quantized = parameters_subtraction(quantized, pairwise_mask)

        # Take mod of final weight update vector and return to server
        return parameters_mod(quantized, self.mod_range)

    # stage 3 ----------------------------------------------------------------------------
    def unmask(self, active_nids: List[int], dead_nids: List[int]) -> Dict[int, bytes]:
        """Send the self-mask seed share of every surviving neighbour (incl. itself) and the
        mask-key share of every dropped neighbour. Returns ``{owner node id: share}``."""
        if len(active_nids) < self.threshold:
            raise ValueError("Available neighbours number smaller than threshold")
        shares = {nid: self.rd_seed_share_dict[nid] for nid in active_nids}
        shares.update({nid: self.sk1_share_dict[nid] for nid in dead_nids})
        return shares


# ----------------------------------------------------------------------------------
# server side  (Flower: flwr/server/workflow/secure_aggregation/secaggplus_workflow.py)

class SecAggPlusServer:
    def __init__(self, params: SecAggPlusParams, node_ids: Sequence[int], rng: Optional[random.Random] = None):
        self.params = params
        self.sampled: Set[int] = set(int(n) for n in node_ids)
        self.num_shares, self.threshold = round_parameters(params, len(self.sampled))
        self.neighbours = build_neighbours(sorted(self.sampled), self.num_shares, rng or random.Random())
        self.public_keys: Dict[int, Tuple[bytes, bytes]] = {}
        self.forward: Dict[int, Dict[int, bytes]] = {nid: {} for nid in self.sampled}
        self.masked_sum: Optional[List[np.ndarray]] = None
        self.collected_shares: Dict[int, List[bytes]] = {}
        # Bonawitz et al.: U1 (sent keys) > U2 (sent shares) > U3 (sent masked input) > U4 (sent unmask shares)
        self.U: Dict[int, Set[int]] = {}
        self.active: Set[int] = set(self.sampled)

    def config(self) -> dict:
        p = self.params
        return {"sample_num": len(self.sampled), "share_num": self.num_shares, "threshold": self.threshold,
                "clipping_range": float(p.clipping_range), "target_range": int(p.quantization_range),
                "mod_range": int(p.modulus_range), "max_weight": float(p.max_weight)}

    def end_stage(self, stage: int, responded: Set[int]) -> None:
        """Record who answered in this stage and run Flower's ``_check_threshold``: every
        client whose secrets may have to be reconstructed (all sampled clients before stage 1
        finished, U2 afterwards) needs at least ``threshold`` active neighbours."""
        self.active = set(responded)
        self.U[stage + 1] = set(responded)
        relevant = self.U.get(2, self.sampled)
        for node_id in sorted(relevant):
            active_neighbours = self.neighbours[node_id] & self.active
            if len(active_neighbours) < self.threshold:
                raise RuntimeError(f"SecAgg+ after stage '{STAGES[stage]}': insufficient available nodes "
                                   f"(client {node_id} has {len(active_neighbours)} active neighbours, "
                                   f"threshold {self.threshold}).")

    # stage 0
    def receive_public_keys(self, nid: int, keys: Tuple[bytes, bytes]) -> None:
        self.public_keys[nid] = (keys[0], keys[1])

    # stage 1
    def public_keys_for(self, nid: int) -> Dict[int, Tuple[bytes, bytes]]:
        return {j: self.public_keys[j] for j in sorted(self.neighbours[nid] & self.active)}

    def receive_ciphertexts(self, src: int, ciphertexts: Dict[int, bytes]) -> None:
        for dst, ct in ciphertexts.items():
            if dst in self.forward:
                self.forward[dst][src] = ct

    # stage 2
    def ciphertexts_for(self, nid: int) -> Dict[int, bytes]:
        return {src: ct for src, ct in self.forward[nid].items() if src in self.active}

    def receive_masked_vector(self, nid: int, masked: List[np.ndarray]) -> None:
        if self.masked_sum is None:
            self.masked_sum = masked
        else:
            self.masked_sum = parameters_addition(self.masked_sum, masked)
        self.masked_sum = parameters_mod(self.masked_sum, self.params.modulus_range)

    # stage 3
    def unmask_request_for(self, nid: int) -> Tuple[List[int], List[int]]:
        neighbours = self.neighbours[nid]
        survivors = self.U[3]
        dropped = self.U[2] - self.U[3]
        return sorted(neighbours & survivors), sorted(neighbours & dropped)

    def receive_unmask_shares(self, nid: int, shares: Dict[int, bytes]) -> None:
        for owner, share in shares.items():
            self.collected_shares.setdefault(owner, []).append(share)

    def reconstruct(self) -> List[np.ndarray]:
        """Remove all masks and return the dequantised weighted average (list of arrays)."""
        p = self.params
        if self.masked_sum is None:
            raise RuntimeError("No masked vectors received.")
        survivors, dropped = self.U[3], self.U[2] - self.U[3]
        masked = self.masked_sum
        shapes = get_parameters_shape(masked)
        for nid in sorted(survivors | dropped):
            share_list = self.collected_shares.get(nid, [])
            if len(share_list) < self.threshold:
                raise RuntimeError(f"Not enough shares to recover the secret of client {nid} "
                                   f"({len(share_list)} < threshold {self.threshold}).")
            secret = combine_shares(share_list)
            if nid in survivors:
                # The seed for PRG is the private mask seed of an active client.
                masked = parameters_subtraction(masked, pseudo_rand_gen(secret, p.modulus_range, shapes))
            else:
                # The seed for PRG is the secret key 1 of a dropped client.
                sk1 = bytes_to_private_key(secret)
                for neighbor_nid in sorted((self.neighbours[nid] - {nid}) & survivors):
                    shared_key = generate_shared_key(sk1, bytes_to_public_key(self.public_keys[neighbor_nid][0]))
                    pairwise_mask = pseudo_rand_gen(shared_key, p.modulus_range, shapes)
                    if nid > neighbor_nid:
                        masked = parameters_addition(masked, pairwise_mask)
                    else:
                        masked = parameters_subtraction(masked, pairwise_mask)
        recon = parameters_mod(masked, p.modulus_range)
        q_total_ratio, recon = factor_extract(recon)
        inv_dq_total_ratio = p.quantization_range / q_total_ratio
        aggregated = dequantize(recon, p.clipping_range, p.quantization_range)
        offset = -(len(survivors) - 1) * p.clipping_range
        for vec in aggregated:
            vec += offset
            vec *= inv_dq_total_ratio
        return aggregated


# ----------  clients -------------------------------
def secaggplus_aggregate(inputs: Dict[int, np.ndarray], params: SecAggPlusParams,
                         weights: Optional[Dict[int, float]] = None,
                         dropouts: Optional[Dict[int, str]] = None,
                         rng: Optional[random.Random] = None, label: str = "secagg") -> Tuple[np.ndarray, dict]:
    """Securely compute the (weighted) average of ``inputs`` (``{client id: 1-D float array}``).

    ``dropouts`` maps a client id to the stage *before which* it disappears:
    ``"share_keys"``, ``"masked_input"`` (the classic case) or ``"unmask"``.
    Returns ``(average, info)``. ``info`` holds the protocol statistics. Communication and
    computation of every stage are recorded with :func:`measure` / :func:`timer` under ``label``.
    """
    dropouts = dict(dropouts or {})
    for nid, stage in dropouts.items():
        if stage not in STAGES[1:]:
            raise ValueError(f"dropout stage of client {nid} must be one of {STAGES[1:]}, got {stage!r}")
    node_ids = sorted(int(n) for n in inputs)
    bits = params.bits_per_entry

    with timer(f"{label}/0 setup [server]"):
        server = SecAggPlusServer(params, node_ids, rng)
    clients = {nid: SecAggPlusClient(nid) for nid in node_ids}
    vectors = {nid: [np.asarray(inputs[nid], dtype=np.float64).reshape(-1)] for nid in node_ids}
    clipped = np.mean([np.mean(np.abs(v[0]) > params.clipping_range) for v in vectors.values()])

    # ---- stage 0: setup --------------------------
    responded = set()
    for nid in node_ids:
        cfg = server.config()
        measure(cfg, f"{label}/0 setup: config", src="server", dst=nid)
        with timer(f"{label}/0 setup [client]", who=nid):
            # each client sets up some config. It sends two public key
            # The first key is for the actual secure communication channel (which we now do here)
            # The second key is for the pairwise KA for s_{i,j} between two FL clients i and j.
            # s_{ij} is used for the model masking. Note that secret shares are send over to respective
            # recieving parties via a secure channel, established through the first public key. 
            keys = clients[nid].setup(cfg)
        measure(keys, f"{label}/0 setup: public keys", src=nid, dst="server")

        # server stores the keys in a datastructure
        server.receive_public_keys(nid, keys)
        responded.add(nid)
    server.end_stage(0, responded)

    # ---- stage 1: in one iteration through all clients we
    # - distributes the pk for the secure channel and the DH key for pairwise s_ij
    # - the clients generate s_ij and b_i and secret share it (with a secure channel) with the server
    responded = set()
    for nid in sorted(server.active):
        # The server forwards the public key. We note that one needs a CA or some
        # other form of trust establishment to properly establish key authenticity.
        keys = server.public_keys_for(nid)
        measure(keys, f"{label}/1 share keys: public keys", src="server", dst=nid)
        if dropouts.get(nid) == "share_keys":
            # client dropped
            continue  

        # client secret shares its key material. Note that it is secured through a secure channel
        with timer(f"{label}/1 share keys [client]", who=nid):
            ciphertexts = clients[nid].share_keys(keys)
        measure(ciphertexts, f"{label}/1 share keys: encrypted shares", src=nid, dst="server")
        # server collects all (encrypted) shares
        server.receive_ciphertexts(nid, ciphertexts)
        responded.add(nid)
    server.end_stage(1, responded)

    # ---- stage 2: in another iteration through all clients we do:
    # - distribute all shares to the clients
    # - and then gather all masked models for the federator
    responded = set()
    for nid in sorted(server.active):
        # distribute shares
        ciphertexts = server.ciphertexts_for(nid)
        measure(ciphertexts, f"{label}/2 masked input: encrypted shares", src="server", dst=nid)
        if dropouts.get(nid) == "masked_input":
            continue
        
        weight = 1.0 if weights is None else float(weights[nid])
        # each client now masks the gradient/model and uploads it to the Fed.
        with timer(f"{label}/2 masked input [client]", who=nid):
            masked = clients[nid].collect_masked_vectors(ciphertexts, vectors[nid], weight)
        measure(masked, f"{label}/2 masked input: masked vector", src=nid, dst="server", bits_per_entry=bits)

        # the Fed. aggregates in-place the model (the resulting model of course is not demasked yet) 
        with timer(f"{label}/2 masked input [server]"):
            server.receive_masked_vector(nid, masked)
        responded.add(nid)
    server.end_stage(2, responded)
 
    # ---- stage 3: unmask ----------------------- 
    # - the server distributes a set of participating clients. 
    # - the clients form a T-quorum for the set of surviving and dropped clients
    # - NOTE: A T-quorum is required for both the sets of b_i and s_ij (for dropped j)
    responded = set()
    for nid in sorted(server.active):

        # inform client nid of the dropped and surviving clients in its neigborhood
        request = server.unmask_request_for(nid)
        measure(request, f"{label}/3 unmask: survivor lists", src="server", dst=nid)
        if dropouts.get(nid) == "unmask":
            continue

        with timer(f"{label}/3 unmask [client]", who=nid):
            # each client now sends its secret sharing parts
            shares = clients[nid].unmask(*request)
        measure(shares, f"{label}/3 unmask: secret shares", src=nid, dst="server")
        # Fed collects shares
        server.receive_unmask_shares(nid, shares)
        responded.add(nid)
    server.end_stage(3, responded)

    # The server reconstructs b_i and s_ij. This is then subtracted from the aggregated masked gradient/model
    # to recover the unmasked aggregate
    with timer(f"{label}/3 unmask [server]"):
        average = server.reconstruct()[0]

    info = {"num_shares": server.num_shares, "threshold": server.threshold,
            "U1": len(server.U[1]), "U2": len(server.U[2]), "U3": len(server.U[3]), "U4": len(server.U[4]),
            "aggregated_clients": sorted(server.U[3]), "clipped_fraction": float(clipped)}
    return average, info


class SecAggPlus:
    """Convenience wrapper used by the experiments::

        secagg = SecAggPlus(num_shares=5, reconstruction_threshold=3)
        mean = secagg.aggregate({cid: vector, ...}, dropouts={3: "masked_input"})

    After a call, ``last_info`` holds the round's statistics (set sizes U1..U4, threshold, ...).
    """

    name = "secagg+"

    def __init__(self, num_shares: float = 1.0, reconstruction_threshold: float = 0.5, max_weight: float = 1.0,
                 clipping_range: float = 8.0, quantization_range: int = 2**22, modulus_range: int = 2**32,
                 seed: Optional[int] = None):
        self.params = SecAggPlusParams(num_shares, reconstruction_threshold, max_weight, clipping_range,
                                       int(quantization_range), int(modulus_range))
        self.rng = random.Random(seed)
        self.last_info: dict = {}

    def aggregate(self, inputs: Dict[int, np.ndarray], weights: Optional[Dict[int, float]] = None,
                  dropouts: Optional[Dict[int, str]] = None) -> np.ndarray:
        average, self.last_info = secaggplus_aggregate(inputs, self.params, weights, dropouts, self.rng,
                                                       label=self.name)
        return average

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"


class SecAgg(SecAggPlus):
    """SecAgg = SecAgg+ on the complete graph (Flower's ``SecAggWorkflow``)."""

    name = "secagg"

    def __init__(self, reconstruction_threshold: float = 0.5, **kwargs):
        kwargs.pop("num_shares", None)
        super().__init__(num_shares=1.0, reconstruction_threshold=reconstruction_threshold, **kwargs)
