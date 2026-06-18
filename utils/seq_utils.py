"""
Sequence utilities for PanCirc-Fungi.

Provides:
- reverse_complement, gc_content, canonical_splice detection
- one_hot_encode, kmer_tokenize for model input preparation
"""

import numpy as np

# ---------------------------------------------------------------------------
# IUPAC nucleotide mapping
# ---------------------------------------------------------------------------
VALID_NT = set("ACGTUacgtu")
COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA/RNA sequence.

    Handles mixed case and RNA U/T ambiguity (U is treated as T).
    """
    return seq.translate(COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    """Fraction of G+C bases. Returns 0.0 for empty sequences."""
    if not seq:
        return 0.0
    s = seq.upper()
    return (s.count("G") + s.count("C")) / len(s)


def canonical_splice(donor: str, acceptor: str) -> bool:
    """Check if acceptor/donor dinucleotides are canonical GT-AG or CT-AC.

    Parameters
    ----------
    donor : str
        Two bases at the 5' (donor) splice site (e.g. 'GT').
    acceptor : str
        Two bases at the 3' (acceptor) splice site (e.g. 'AG').

    Returns
    -------
    bool
    """
    d = donor.upper().strip()
    a = acceptor.upper().strip()
    return (d == "GT" and a == "AG") or (d == "CT" and a == "AC")


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------
_NT_TO_ONEHOT = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "U": [0.0, 0.0, 0.0, 1.0],
    # Ambiguous: uniform distribution
    "N": [0.25, 0.25, 0.25, 0.25],
    "R": [0.5, 0.0, 0.5, 0.0],
    "Y": [0.0, 0.5, 0.0, 0.5],
    "S": [0.0, 0.5, 0.5, 0.0],
    "W": [0.5, 0.0, 0.0, 0.5],
    "M": [0.5, 0.5, 0.0, 0.0],
    "K": [0.0, 0.0, 0.5, 0.5],
    "B": [0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3],
    "D": [1.0 / 3, 0.0, 1.0 / 3, 1.0 / 3],
    "H": [1.0 / 3, 1.0 / 3, 0.0, 1.0 / 3],
    "V": [1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0],
}


def one_hot_encode(seq: str, order: str = "ACGT") -> np.ndarray:
    """One-hot encode a DNA/RNA sequence.

    Returns array of shape (4, L), where L = len(seq).
    Order can be changed via the ``order`` parameter.
    """
    order = order.upper()
    lookup = {c: i for i, c in enumerate(order)}
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    for i, nt in enumerate(seq.upper()):
        if nt in lookup:
            arr[lookup[nt], i] = 1.0
        else:
            # Ambiguous or N: distribute equally
            arr[:, i] = _NT_TO_ONEHOT.get(nt, [0.25, 0.25, 0.25, 0.25])
    return arr


# ---------------------------------------------------------------------------
# k-mer tokenization
# ---------------------------------------------------------------------------
def _kmer_table(k: int) -> dict:
    """Build lookup: k-mer string -> integer index [0, 4^k)."""
    bases = "ACGT"
    table = {}
    for i in range(4**k):
        kmer = ""
        tmp = i
        for _ in range(k):
            kmer = bases[tmp % 4] + kmer
            tmp //= 4
        table[kmer] = i
    return table


_KMER_CACHE: dict[int, dict] = {}


def kmer_tokenize(seq: str, k: int = 3, stride: int = 1) -> np.ndarray:
    """Convert a sequence to integer k-mer token IDs.

    Returns array of length (len(seq) - k + 1) // stride.
    Ambiguous bases are mapped to the closest k-mer (first base determines).
    """
    if k not in _KMER_CACHE:
        _KMER_CACHE[k] = _kmer_table(k)
    table = _KMER_CACHE[k]

    seq = seq.upper().replace("U", "T")
    tokens = []
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i : i + k]
        # Replace non-standard bases with 'A' as fallback
        clean = "".join(c if c in "ACGT" else "A" for c in kmer)
        tokens.append(table.get(clean, 0))
    return np.array(tokens, dtype=np.int64)
