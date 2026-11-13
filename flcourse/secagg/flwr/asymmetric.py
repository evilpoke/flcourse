# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# ------------------------------------------------------------------------------
# Vendored into flcourse from Flower 1.42.0 (https://github.com/flwrlabs/flower),
# file: flwr/supercore/primitives/asymmetric.py
# Changes by flcourse: subset of the file (key generation and (de)serialisation
# only; the signature helpers and their imports were dropped).
# ------------------------------------------------------------------------------
"""Asymmetric cryptography utilities."""


from typing import cast

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def generate_key_pairs() -> (
    tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
):
    """Generate private and public key pairs with Cryptography."""
    private_key = ec.generate_private_key(ec.SECP384R1())
    public_key = private_key.public_key()
    return private_key, public_key


def private_key_to_bytes(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    """Serialize private key to bytes."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def bytes_to_private_key(private_key_bytes: bytes) -> ec.EllipticCurvePrivateKey:
    """Deserialize private key from bytes."""
    return cast(
        ec.EllipticCurvePrivateKey,
        serialization.load_pem_private_key(data=private_key_bytes, password=None),
    )


def public_key_to_bytes(public_key: ec.EllipticCurvePublicKey) -> bytes:
    """Serialize public key to bytes."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def bytes_to_public_key(public_key_bytes: bytes) -> ec.EllipticCurvePublicKey:
    """Deserialize public key from bytes."""
    return cast(
        ec.EllipticCurvePublicKey,
        serialization.load_pem_public_key(data=public_key_bytes),
    )
