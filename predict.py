from Bio.SeqUtils import gc_frequency
import re
import numpy as np
# ==================== 特征提取模块 ====================
class SARSCoV2EfficiencyPredictor:
    def __init__(self):
        # 定义模板转换核心序列（根据[1](@ref)）
        self.trs_core = "ACGAAC"  # 典型TRS motif
        # 人类密码子使用表（简化版，基于[2](@ref)）
        self.human_codon_table = {
            'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,  # Ala
            'TGT': 0.45, 'TGC': 0.55,  # Cys
            'GAT': 0.46, 'GAC': 0.54,  # Asp
            'GAA': 0.42, 'GAG': 0.58,  # Glu
            'TTT': 0.45, 'TTC': 0.55,  # Phe
            'GGT': 0.35, 'GGC': 0.45, 'GGA': 0.12, 'GGG': 0.08,  # Gly
            'CAT': 0.41, 'CAC': 0.59,  # His
            'ATT': 0.36, 'ATC': 0.48, 'ATA': 0.16,  # Ile
            'AAA': 0.43, 'AAG': 0.57,  # Lys
            'TTA': 0.14, 'TTG': 0.13, 'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.33,  # Leu
            'ATG': 1.0,  # Met
            'AAT': 0.46, 'AAC': 0.54,  # Asn
            'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.12,  # Pro
            'CAA': 0.26, 'CAG': 0.74,  # Gln
            'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21,  # Arg
            'TCT': 0.19, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.06, 'AGT': 0.15, 'AGC': 0.23,  # Ser
            'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,  # Thr
            'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47,  # Val
            'TGG': 1.0,  # Trp
            'TAT': 0.43, 'TAC': 0.57,  # Tyr
            'TAA': 0.30, 'TAG': 0.24, 'TGA': 0.46  # Stop
        }
        # RdRp保守模体（基于[1](@ref)和[4](@ref)）
        self.rdrp_motifs = [
            r"G..T.SP.VW",  # 催化核心模体
            r"RxP.L.D.G",  # GSD模体
            r"LDSW.I.E.LD"  # 模板结合区域
        ]

    def count_trs_sites(self, sequence):
        """计算TRS位点频率（基于[1](@ref)的模板转换特征）"""
        sites = len(re.findall(self.trs_core, sequence.upper()))
        return sites / (len(sequence) / 1000)  # 归一化为每千碱基

    def calculate_cai(self, sequence):
        """计算密码子适应指数（CAI），反映密码子使用优化程度[2](@ref)"""
        cds_list = self._extract_orfs(sequence)
        if not cds_list:
            return 0.0
        cai_values = []
        for cds in cds_list:
            codon_counts = {}
            for i in range(0, len(cds), 3):
                codon = cds[i:i + 3]
                if len(codon) == 3 and 'N' not in codon:
                    codon_counts[codon] = codon_counts.get(codon, 0) + 1
            total_weight = 0
            total_codons = sum(codon_counts.values())
            for codon, count in codon_counts.items():
                if codon in self.human_codon_table:
                    total_weight += count * self.human_codon_table[codon]
            cai = total_weight / total_codons if total_codons > 0 else 0
            cai_values.append(cai)
        return np.mean(cai_values)

    @staticmethod
    def _extract_orfs(self, sequence, min_length=300):
        """提取开放阅读框（简化版）"""
        orfs = []
        for match in re.finditer(r'ATG(?:\w{3})*?(?:TAA|TAG|TGA)', sequence.upper()):
            orf = match.group()
            if len(orf) >= min_length:
                orfs.append(orf)
        return orfs

    def check_conserved_motifs(self, sequence):
        """检测RdRp等关键蛋白的保守模体完整性[1,4](@ref)"""
        score = 0
        for motif in self.rdrp_motifs:
            if re.search(motif, sequence.upper()):
                score += 1
        return score / len(self.rdrp_motifs)  # 返回保守比例

    @staticmethod
    def calculate_gc_stability(self, sequence):
        """通过GC含量评估RNA稳定性[1](@ref)"""
        gc = gc_frequency(sequence)
        # GC含量在40%-50%时稳定性最佳[1](@ref)
        return 1 - abs(gc - 45) / 45  # 归一化到0-1

    # ==================== 效率预测模块 ====================
    def predict_efficiency(self, sequence):
        """综合预测复制与转录效率"""
        features = {
            'trs_density': self.count_trs_sites(sequence),
            'cai': self.calculate_cai(sequence),
            'motif_integrity': self.check_conserved_motifs(sequence),
            'gc_stability': self.calculate_gc_stability(sequence)
        }
        # 特征权重（基于文献启发式设置）
        weights = {
            'trs_density': 0.4,  # TRS频率对转录效率影响大[1](@ref)
            'cai': 0.3,  # 密码子优化影响翻译效率[2](@ref)
            'motif_integrity': 0.2,  # 关键酶完整性影响复制[4](@ref)
            'gc_stability': 0.1  # RNA稳定性次要因素
        }
        # 归一化特征
        trs_norm = min(features['trs_density'] / 10, 1.0)  # 假设每千碱基≤10为饱和
        cai_norm = features['cai']
        motif_norm = features['motif_integrity']
        gc_norm = features['gc_stability']

        # 计算综合分数
        score = (weights['trs_density'] * trs_norm +
                 weights['cai'] * cai_norm +
                 weights['motif_integrity'] * motif_norm +
                 weights['gc_stability'] * gc_norm)

        # 转录效率侧重TRS和CAI，复制效率侧重模体和稳定性
        transcription_score = (0.6 * trs_norm + 0.4 * cai_norm)
        replication_score = (0.5 * motif_norm + 0.3 * gc_norm + 0.2 * cai_norm)

        return {
            'combined_efficiency': round(score, 4),
            'transcription_efficiency': round(transcription_score, 4),
            'replication_efficiency': round(replication_score, 4),
            'feature_details': features
        }


# ==================== 示例使用 ====================
if __name__ == "__main__":
    # 示例序列（替换为实际新冠病毒基因组，如NC_045512）
    example_sequence = "TAAAGGTTTATACCTTCCTAGGTAACAAACCAACCAACTTTTGATCTCTTGTAGATCTGTTCTCTAAACGAACTTTAAAATCTGTGTGGCTGTCACTCGGCTGCATGCTTAGTGCACTCACGCAGTATAATTAATAACTAATTACTGTCGTTGACAGGACACGAGTAACTCGTCTATCTTCTGCAGGCTGCTTACGGTTTCGTCCGTGTTGCAGCCGATCATCAGCACATCTAGGTTTTGTCCGGGTGTGACCGAAAGGTAAGATGGAGAGCCTTGTCCCTGGTTTCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGTGCTCGTACGTGGCTTTGGAGACTCCGTGGAGGAGGTCTTATCAGAGGCACGTCAACATCTTAAAGATGGCACTTGTGGCTTAGTAGAAGTTGAAAAAGGCGTTTTGCCTCAACTTGAACAGCCCTATGTGTTCATCAAACGTTCGGATGCTCGAACTGCACCTCATGGTCATGTTATGGTTGAGCTGGTAGCAGAACTCGAAGGCATTCAGTACGGTCGTAGTGGTGAGACACTTGGTGTCCTTGTCCCTCATGTGGGCGAAATACCAGTGGCTTACCGCAAGGTTCTTCTTCGTAAGAACGGTAATAAAGGAGCTGGTGGCCATAGGTACGGCGCCGATCTAAAGTCATTTGACTTAGGCGACGAGCTTGGCACTGATCCTTATGAAGATTTTCAAGAAAACTGGAACACTAAACATAGCAGTGGTGTTACCCGTGAACTCATGCGTGAGCTTAACGGAGGGGCATACACTCGCTATGTCGATAACAACTTCTGTGGCC"
    # 此处应输入完整序列

    predictor = SARSCoV2EfficiencyPredictor()
    result = predictor.predict_efficiency(example_sequence)

    print("=== 新冠病毒复制/转录效率预测结果 ===")
    print(f"综合效率分数: {result['combined_efficiency']}")
    print(f"转录效率: {result['transcription_efficiency']}")
    print(f"复制效率: {result['replication_efficiency']}")
    print("\n--- 特征详情 ---")
    for feature, value in result['feature_details'].items():
        print(f"{feature}: {value:.4f}")
