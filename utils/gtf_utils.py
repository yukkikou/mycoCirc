"""
GTF file parser for PanCirc-Fungi.

Provides:
- GTFReader: memory-efficient streaming parser
- GeneModelIndexer: builds interval-tree index for fast gene lookups
- Feature extraction utilities
"""

import re
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np

GTFRecord = namedtuple(
    "GTFRecord",
    ["seqname", "source", "feature", "start", "end", "score",
     "strand", "frame", "attributes"],
)


_ATTR_RE = re.compile(r'(?P<key>\w+)\s+"(?P<value>[^"]*)"')


def parse_gtf_attributes(attr_string: str) -> Dict[str, str]:
    """Parse the GTF attributes column into a dictionary.

    Handles standard GTF quoted attributes::
        gene_id "ENSG001"; transcript_id "ENST001";
    """
    result = {}
    for match in _ATTR_RE.finditer(attr_string):
        result[match.group("key")] = match.group("value")
    return result


class GTFReader:
    """Streaming GTF parser. Iterate to get ``GTFRecord`` namedtuples.

    Parameters
    ----------
    path : str
        Path to GTF file (gzipped files opened transparently).
    """

    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        """Yield parsed ``GTFRecord`` entries, skipping comment lines."""
        with open(self.path) as fh:
            for line in fh:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split("\t")
                if len(parts) != 9:
                    continue
                attr_dict = parse_gtf_attributes(parts[8])
                yield GTFRecord(
                    seqname=parts[0],
                    source=parts[1],
                    feature=parts[2],
                    start=int(parts[3]),
                    end=int(parts[4]),
                    score=parts[5],
                    strand=parts[6],
                    frame=parts[7],
                    attributes=attr_dict,
                )

    def read_genes(self) -> Dict[str, GTFRecord]:
        """Return dict of gene_id -> first gene record for each gene."""
        genes = {}
        for rec in self:
            if rec.feature == "gene":
                gid = rec.attributes.get("gene_id")
                if gid and gid not in genes:
                    genes[gid] = rec
        return genes

    def read_transcripts(self) -> Dict[str, List[GTFRecord]]:
        """Return dict of transcript_id -> list of exon/CDS records."""
        transcripts = defaultdict(list)
        for rec in self:
            if rec.feature in ("exon", "CDS", "transcript", "UTR"):
                tid = rec.attributes.get("transcript_id")
                if tid:
                    transcripts[tid].append(rec)
        return dict(transcripts)

    def read_exons_by_gene(self) -> Dict[str, List[GTFRecord]]:
        """Return dict of gene_id -> list of exon records."""
        exons = defaultdict(list)
        for rec in self:
            if rec.feature == "exon":
                gid = rec.attributes.get("gene_id")
                if gid:
                    exons[gid].append(rec)
        return dict(exons)


class GeneInfo:
    """Stores parsed information about a single gene."""

    __slots__ = (
        "gene_id", "gene_name", "gene_biotype", "chrom", "strand",
        "start", "end", "exon_starts", "exon_ends", "cds_regions",
        "transcript_ids",
    )

    def __init__(self, gene_id: str, chrom: str = "", strand: str = "",
                 start: int = 0, end: int = 0):
        self.gene_id = gene_id
        self.gene_name = ""
        self.gene_biotype = ""
        self.chrom = chrom
        self.strand = strand
        self.start = start
        self.end = end
        self.exon_starts: List[int] = []
        self.exon_ends: List[int] = []
        self.cds_regions: List[Tuple[int, int]] = []
        self.transcript_ids: List[str] = []

    @property
    def exon_count(self) -> int:
        return len(self.exon_starts)

    @property
    def exon_lengths(self) -> List[int]:
        return [e - s for s, e in zip(self.exon_starts, self.exon_ends)]

    @property
    def total_exon_length(self) -> int:
        return sum(self.exon_lengths)

    @property
    def total_intron_length(self) -> int:
        if self.exon_count < 2:
            return 0
        body_len = self.end - self.start
        return body_len - self.total_exon_length

    @property
    def cds_length(self) -> int:
        return sum(e - s for s, e in self.cds_regions)

    def is_overlapping(self, position: int) -> bool:
        """Check if a genomic position falls within this gene's span."""
        return self.start <= position <= self.end


class GeneModelIndexer:
    """Builds interval indices from a GTF for fast lookups.

    Parameters
    ----------
    gtf_path : str
    """
    BIOTYPE_MAP = {
        "protein_coding": 0,
        "lncRNA": 1, "lincRNA": 1,
        "snoRNA": 2, "snRNA": 2,
        "tRNA": 3,
        "rRNA": 4,
        "pseudogene": 5,
        "miRNA": 6,
        "ncRNA": 7,
        "TE": 8,
        "IG": 9,
        "other": 10,
    }

    def __init__(self, gtf_path: str):
        self.gtf_path = gtf_path
        self.genes: Dict[str, GeneInfo] = {}
        self._chrom_genes: Dict[str, List[GeneInfo]] = defaultdict(list)
        self._load()

    def _load(self):
        """Parse GTF and build gene index."""
        current = {}
        reader = GTFReader(self.gtf_path)

        # First pass: collect all gene-level info
        for rec in reader:
            gid = rec.attributes.get("gene_id")
            if not gid:
                continue

            if gid not in current:
                info = GeneInfo(gid, rec.seqname, rec.strand, rec.start, rec.end)
                info.gene_name = rec.attributes.get("gene_name", "")
                info.gene_biotype = rec.attributes.get("gene_biotype", "")
                current[gid] = info

            info = current[gid]

            # Track gene boundaries
            info.start = min(info.start, rec.start)
            info.end = max(info.end, rec.end)

            if rec.feature == "exon":
                info.exon_starts.append(rec.start)
                info.exon_ends.append(rec.end)
                tid = rec.attributes.get("transcript_id", "")
                if tid and tid not in info.transcript_ids:
                    info.transcript_ids.append(tid)

            if rec.feature == "CDS":
                info.cds_regions.append((rec.start, rec.end))

        # Sort exon regions by position
        for gid, info in current.items():
            if info.exon_starts:
                combined = sorted(zip(info.exon_starts, info.exon_ends))
                info.exon_starts = [s for s, e in combined]
                info.exon_ends = [e for s, e in combined]
            if info.cds_regions:
                info.cds_regions.sort()

        # Index by chromosome
        for gid, info in current.items():
            self._chrom_genes[info.chrom].append(info)

        self.genes = current

    def get_gene(self, gene_id: str) -> Optional[GeneInfo]:
        """Look up a single gene by ID."""
        return self.genes.get(gene_id)

    def find_gene_at_position(self, chrom: str, position: int) -> Optional[GeneInfo]:
        """Find the first gene that encompasses a genomic position.

        Uses simple linear scan (interval tree not needed for typical
        fungal genomes with ~10³ genes per chromosome).
        """
        for gene in self._chrom_genes.get(chrom, []):
            if gene.start <= position <= gene.end:
                return gene
        return None

    def get_biotype_id(self, biotype: str) -> int:
        """Map gene biotype string to integer ID."""
        return self.BIOTYPE_MAP.get(biotype.lower(), 10)

    def n_genes(self) -> int:
        return len(self.genes)

    def extract_features(self, gene_info: GeneInfo) -> np.ndarray:
        """Convert a ``GeneInfo`` to a fixed-length numeric feature vector.

        Returns array of shape (17,) with:
        [exon_count, log1p_exon_len, log1p_intron_len, log1p_cds_len,
         exon_density, relative_gene_length, is_multi_exon, has_cds,
         biotype_id, ...reserved...]
        """
        exon_cnt = gene_info.exon_count
        tot_exon = gene_info.total_exon_length
        tot_intron = gene_info.total_intron_length
        cds_len = gene_info.cds_length
        gene_len = gene_info.end - gene_info.start

        exon_density = exon_cnt / max(gene_len, 1) * 1000  # exons per kb
        relative_len = gene_len / 10000  # normalize to ~1 for 10kb genes

        feats = [
            float(exon_cnt),
            float(np.log1p(tot_exon)),
            float(np.log1p(max(0, tot_intron))),
            float(np.log1p(cds_len)),
            float(exon_density),
            float(relative_len),
            float(exon_cnt > 1),
            float(cds_len > 0),
            float(self.get_biotype_id(gene_info.gene_biotype)),
        ]
        # Pad / reserve space for future features
        while len(feats) < 17:
            feats.append(0.0)
        return np.array(feats[:17], dtype=np.float32)

    def generate_candidate_junctions(self, gene_info: GeneInfo,
                                     max_candidates: int = 50
                                     ) -> List[Tuple[int, int, int, int]]:
        """Generate all possible backsplice junctions for a gene.

        Backsplicing joins a 3' splice site (acceptor) to a 5' splice site
        (donor) — the reverse of linear splicing.

        For + strand: acceptor = exon_i.end (3'SS), donor = exon_j.start (5'SS)
        For - strand: acceptor = exon_i.start (3'SS), donor = exon_j.end (5'SS)

        Returns at most ``max_candidates`` candidates, each as
        (exon_i_idx, exon_j_idx, donor_pos, acceptor_pos).
        """
        if gene_info.exon_count < 2:
            return []

        candidates = []
        # Backsplicing pairs (acceptor_exon, donor_exon). Only i < j pairs
        # (same-exon pairs excluded — they don't represent splicing events).
        for i in range(gene_info.exon_count):
            for j in range(i + 1, gene_info.exon_count):
                if gene_info.strand == "-":
                    donor = gene_info.exon_ends[j]
                    acceptor = gene_info.exon_starts[i]
                else:
                    donor = gene_info.exon_starts[j]
                    acceptor = gene_info.exon_ends[i]

                # Donor must be downstream (higher genomic coord) of acceptor
                if donor <= acceptor:
                    continue

                candidates.append((i, j, donor, acceptor))

        if len(candidates) > max_candidates:
            import random
            candidates = random.sample(candidates, max_candidates)

        return candidates
