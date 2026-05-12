"""
 AHP 交互式分析工具（数模竞赛专用）
 运行: python ahp_interactive.py
 依赖: numpy (必需)，matplotlib (可视化时可选)
"""
import numpy as np
import sys
import os

# Saaty 随机一致性指标 RI 表
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
            11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}

"""
# ============================================================================
#                           输入辅助工具
# ============================================================================
"""
def parse_number(s):
    """解析数字，支持分数 1/3"""
    s = s.strip()
    if '/' in s:
        a, b = s.split('/')
        return float(a) / float(b)
    return float(s)


def ask_int(prompt, default=None, min_val=1, max_val=15):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            v = int(raw)
            if min_val <= v <= max_val:
                return v
            print(f"  ⚠ 范围应为 {min_val}~{max_val}")
        except ValueError:
            print("  ⚠ 请输入整数")


def ask_yes_no(prompt, default='y'):
    raw = input(f"{prompt} (y/n) [{default}]: ").strip().lower()
    if not raw:
        raw = default
    return raw in ('y', 'yes', '是')


def display_matrix(matrix, labels):
    w = max(max(len(str(l)) for l in labels) + 2, 10)
    print(" " * w + "".join(f"{l:>{w}}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<{w}}" + "".join(f"{matrix[i, j]:>{w}.4f}"
                                     for j in range(len(labels))))


def input_matrix_upper(n, labels):
    """上三角输入 (推荐)"""
    print(f"\n>>> 上三角输入: 共需 {n*(n-1)//2} 个数")
    print(">>> Saaty 标度: 1(同等) 3(稍重) 5(明显) 7(强烈) 9(极端)")
    print(">>> 支持分数: 1/3 等\n")
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            while True:
                try:
                    raw = input(f"  {labels[i]} ⇆ {labels[j]} (前者相对后者): ").strip()
                    val = parse_number(raw)
                    if val <= 0:
                        print("    ⚠ 必须为正数")
                        continue
                    matrix[i, j] = val
                    matrix[j, i] = 1.0 / val
                    break
                except (ValueError, ZeroDivisionError):
                    print("    ⚠ 格式错误，可输入: 3, 0.5, 1/3 等")
    return matrix


def input_matrix_full(n, labels):
    """完整输入"""
    print(f"\n>>> 完整输入: 每行 {n} 个数，空格分隔，支持分数\n")
    matrix = np.zeros((n, n))
    for i in range(n):
        while True:
            line = input(f"  第 {i+1} 行 ({labels[i]}): ").split()
            if len(line) != n:
                print(f"    ⚠ 需要 {n} 个元素")
                continue
            try:
                matrix[i] = [parse_number(x) for x in line]
                if (matrix[i] <= 0).any():
                    print("    ⚠ 元素必须为正")
                    continue
                break
            except (ValueError, ZeroDivisionError):
                print("    ⚠ 格式错误")
    return matrix


def input_matrix_interactive(n, labels, title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")
    while True:
        if ask_yes_no("使用「上三角输入」(推荐)？", 'y'):
            m = input_matrix_upper(n, labels)
        else:
            m = input_matrix_full(n, labels)
        print("\n您输入的矩阵为：")
        display_matrix(m, labels)
        if ask_yes_no("\n确认无误？", 'y'):
            return m
        print("\n好，重新输入...\n")


# 
#                           AHP 核心计算

class AHPMatrix:
    def __init__(self, matrix, name="判断矩阵"):
        self.matrix = np.array(matrix, dtype=float)
        self.name = name
        self.n = self.matrix.shape[0]
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"[{name}] 必须为方阵")
        if (self.matrix <= 0).any():
            raise ValueError(f"[{name}] 元素必须为正数")
        self._compute()

    def _compute(self):
        normalized = self.matrix / self.matrix.sum(axis=0)
        self.w_arith = normalized.mean(axis=1)

        geo = np.prod(self.matrix, axis=1) ** (1.0 / self.n)
        self.w_geo = geo / geo.sum()

        eigvals, eigvecs = np.linalg.eig(self.matrix)
        idx = np.argmax(eigvals.real)
        self.lambda_max = eigvals[idx].real
        w = eigvecs[:, idx].real
        if w.sum() < 0:
            w = -w
        self.w_eigen = w / w.sum()

        if self.n <= 2:
            self.CI, self.RI, self.CR = 0.0, 0.0, 0.0
        else:
            self.CI = (self.lambda_max - self.n) / (self.n - 1)
            self.RI = RI_TABLE.get(self.n, 1.59)
            self.CR = self.CI / self.RI if self.RI > 0 else 0.0
        self.is_consistent = self.CR < 0.1

    def get_weights(self, method='eigen'):
        return {'eigen': self.w_eigen,
                'arithmetic': self.w_arith,
                'geometric': self.w_geo}[method]


class AHPModel:
    METHOD_NAME = {'eigen': '特征值法',
                   'arithmetic': '算术平均法',
                   'geometric': '几何平均法'}

    def __init__(self, criteria_matrix, alternative_matrices,
                 criteria_names=None, alternative_names=None,
                 goal_name="决策目标", method='eigen'):
        self.goal_name = goal_name
        self.method = method
        self.criteria_ahp = AHPMatrix(criteria_matrix, "准则层")
        n = self.criteria_ahp.n
        self.criteria_names = criteria_names or [f"C{i+1}" for i in range(n)]
        if len(alternative_matrices) != n:
            raise ValueError(f"方案矩阵数({len(alternative_matrices)})≠准则数({n})")
        self.alt_ahps = [AHPMatrix(m, f"方案-{self.criteria_names[i]}")
                         for i, m in enumerate(alternative_matrices)]
        m_alts = self.alt_ahps[0].n
        self.alternative_names = alternative_names or [f"A{i+1}" for i in range(m_alts)]
        self._compute_total()

    def _compute_total(self):
        self.w_criteria = self.criteria_ahp.get_weights(self.method)
        n = len(self.criteria_names)
        m = len(self.alternative_names)
        W = np.zeros((m, n))
        for j, ahp in enumerate(self.alt_ahps):
            W[:, j] = ahp.get_weights(self.method)
        self.W_alt = W
        self.total_weights = W @ self.w_criteria
        self.ranking_idx = np.argsort(-self.total_weights)
        self.ranking = [self.alternative_names[i] for i in self.ranking_idx]
        self.ranks = np.argsort(np.argsort(-self.total_weights)) + 1
        CIs = np.array([a.CI for a in self.alt_ahps])
        RIs = np.array([a.RI for a in self.alt_ahps])
        num = (self.w_criteria * CIs).sum()
        den = (self.w_criteria * RIs).sum()
        self.total_CR = num / den if den > 0 else 0.0
        self.total_consistent = self.total_CR < 0.1

    # ─── 终端报告 ───
    def print_report(self):
        line = lambda c='─', n=70: c * n
        print(f"\n{line('═')}")
        print(f"  AHP 完整分析报告")
        print(f"  目标：{self.goal_name}")
        print(f"  方法：{self.METHOD_NAME[self.method]}")
        print(line('═'))
        self._print_matrix(self.criteria_ahp, self.criteria_names, "一、准则层分析")
        for i, ahp in enumerate(self.alt_ahps):
            self._print_matrix(ahp, self.alternative_names,
                               f"二.{i+1} 方案对「{self.criteria_names[i]}」")
        print(f"\n{line('═')}\n  三、层次总排序\n{line('═')}")
        hdr = f"{'方案':<10}"
        for i, name in enumerate(self.criteria_names):
            hdr += f"{name}({self.w_criteria[i]:.3f})".rjust(16)
        hdr += "综合权重".rjust(12) + "排名".rjust(8)
        print(hdr)
        print('─' * (10 + 16 * len(self.criteria_names) + 20))
        for i, alt in enumerate(self.alternative_names):
            row = f"{alt:<10}"
            for j in range(len(self.criteria_names)):
                row += f"{self.W_alt[i, j]:.4f}".rjust(16)
            row += f"{self.total_weights[i]:.4f}".rjust(12)
            row += f"{self.ranks[i]}".rjust(8)
            print(row)
        print(f"\n层次总排序一致性比率 CR = {self.total_CR:.4f}")
        print("✓ 通过总一致性检验" if self.total_consistent
              else "✗ 未通过总一致性检验！")
        print(f"\n{line('═')}\n  四、结论\n{line('═')}")
        print(f"  最优方案：{self.ranking[0]}  "
              f"(综合权重 = {self.total_weights[self.ranking_idx[0]]:.4f})")
        print(f"  完整排序：{' > '.join(self.ranking)}\n")

    def _print_matrix(self, ahp, labels, title):
        print(f"\n{'─'*60}\n  {title}\n{'─'*60}")
        print("\n判断矩阵：")
        display_matrix(ahp.matrix, labels)
        print(f"\n权重 (三种方法对比)：")
        print(f"{'因素':<10}{'算术平均':>14}{'几何平均':>14}{'特征值法':>14}")
        for i, l in enumerate(labels):
            print(f"{l:<10}{ahp.w_arith[i]:>14.4f}"
                  f"{ahp.w_geo[i]:>14.4f}{ahp.w_eigen[i]:>14.4f}")
        print(f"\n一致性检验：")
        print(f"  λ_max = {ahp.lambda_max:.4f}, CI = {ahp.CI:.4f}, "
              f"RI = {ahp.RI:.4f}, CR = {ahp.CR:.4f}")
        print("  ✓ 通过" if ahp.is_consistent else "  ✗ 未通过！请调整矩阵")

    # ─── Markdown 输出 ───
    def to_markdown(self):
        L = [f"## AHP 分析报告：{self.goal_name}", "",
             f"**权重计算方法**：{self.METHOD_NAME[self.method]}", "",
             "### 一、准则层分析", "",
             self._md_matrix(self.criteria_ahp, self.criteria_names, "准则层判断矩阵"),
             "", self._md_weights(self.criteria_ahp, self.criteria_names), "",
             self._md_consistency(self.criteria_ahp), "",
             "### 二、方案层分析", ""]
        for i, ahp in enumerate(self.alt_ahps):
            L += [f"#### 2.{i+1} 方案对「{self.criteria_names[i]}」", "",
                  self._md_matrix(ahp, self.alternative_names, ""), "",
                  self._md_weights(ahp, self.alternative_names), "",
                  self._md_consistency(ahp), ""]
        L += ["### 三、层次总排序", ""]
        hdr = "| 方案 |"
        sep = "|---|"
        for i, n in enumerate(self.criteria_names):
            hdr += f" {n} ({self.w_criteria[i]:.3f}) |"
            sep += "---|"
        hdr += " **综合权重** | 排名 |"
        sep += "---|---|"
        L += [hdr, sep]
        for i, alt in enumerate(self.alternative_names):
            row = f"| {alt} |"
            for j in range(len(self.criteria_names)):
                row += f" {self.W_alt[i, j]:.4f} |"
            row += f" **{self.total_weights[i]:.4f}** | {self.ranks[i]} |"
            L.append(row)
        L += ["", f"**总排序一致性比率** CR = {self.total_CR:.4f} "
              f"{'✓ 通过' if self.total_consistent else '✗ 未通过'}", "",
              "### 四、结论", "",
              f"- **最优方案**: **{self.ranking[0]}** "
              f"(综合权重 = {self.total_weights[self.ranking_idx[0]]:.4f})",
              f"- **完整排序**: {' > '.join(self.ranking)}", ""]
        return "\n".join(L)

    def _md_matrix(self, ahp, labels, title):
        out = [f"**{title}**", ""] if title else []
        out += ["|  | " + " | ".join(labels) + " |",
                "|---|" + "---|" * len(labels)]
        for i, l in enumerate(labels):
            out.append(f"| **{l}** | " + " | ".join(
                f"{ahp.matrix[i, j]:.4f}" for j in range(ahp.n)) + " |")
        return "\n".join(out)

    def _md_weights(self, ahp, labels):
        out = ["**权重计算结果**", "",
               "| 因素 | 算术平均法 | 几何平均法 | 特征值法 |",
               "|---|---|---|---|"]
        for i, l in enumerate(labels):
            out.append(f"| {l} | {ahp.w_arith[i]:.4f} | "
                       f"{ahp.w_geo[i]:.4f} | {ahp.w_eigen[i]:.4f} |")
        return "\n".join(out)

    def _md_consistency(self, ahp):
        flag = "✓ 通过" if ahp.is_consistent else "✗ 未通过"
        return (f"**一致性检验**: λ_max = {ahp.lambda_max:.4f}, "
                f"CI = {ahp.CI:.4f}, RI = {ahp.RI:.4f}, "
                f"**CR = {ahp.CR:.4f}** {flag}")

    # ─── LaTeX 输出 ───
    def to_latex(self):
        L = [r"% ===== AHP LaTeX 报告（需 \usepackage{float}）=====", ""]
        L += [self._latex_matrix(self.criteria_ahp, self.criteria_names,
                                 "准则层判断矩阵"), ""]
        for i, ahp in enumerate(self.alt_ahps):
            L += [self._latex_matrix(ahp, self.alternative_names,
                                     f"方案层对「{self.criteria_names[i]}」判断矩阵"), ""]
        L += [self._latex_consistency_table(), "",
              self._latex_final_table(), ""]
        return "\n".join(L)

    def _latex_matrix(self, ahp, labels, caption):
        col = "c|" + "c" * ahp.n
        out = [r"\begin{table}[H]", r"\centering",
               f"\\caption{{{caption}}}",
               f"\\begin{{tabular}}{{{col}}}", r"\hline",
               " & " + " & ".join(labels) + r" \\", r"\hline"]
        for i, l in enumerate(labels):
            row = l + " & " + " & ".join(f"{ahp.matrix[i, j]:.4f}"
                                          for j in range(ahp.n))
            out.append(row + r" \\")
        out += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(out)

    def _latex_consistency_table(self):
        out = [r"\begin{table}[H]", r"\centering",
               r"\caption{各判断矩阵一致性检验}",
               r"\begin{tabular}{l|cccc|c}", r"\hline",
               r"判断矩阵 & $\lambda_{max}$ & CI & RI & CR & 一致性 \\", r"\hline"]
        all_ahps = [("准则层", self.criteria_ahp)] + \
                   [(f"方案-{n}", a) for n, a in zip(self.criteria_names, self.alt_ahps)]
        for name, a in all_ahps:
            flag = "通过" if a.is_consistent else "不通过"
            out.append(f"{name} & {a.lambda_max:.4f} & {a.CI:.4f} & "
                       f"{a.RI:.4f} & {a.CR:.4f} & {flag} \\\\")
        out += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(out)

    def _latex_final_table(self):
        n = len(self.criteria_names)
        col = "c|" + "c" * n + "|cc"
        out = [r"\begin{table}[H]", r"\centering",
               r"\caption{层次总排序结果}",
               f"\\begin{{tabular}}{{{col}}}", r"\hline"]
        hdr = "方案 & " + " & ".join(
            f"{name}({self.w_criteria[i]:.3f})"
            for i, name in enumerate(self.criteria_names)) + r" & 综合权重 & 排名 \\"
        out += [hdr, r"\hline"]
        for i, alt in enumerate(self.alternative_names):
            row = alt + " & " + " & ".join(f"{self.W_alt[i, j]:.4f}" for j in range(n))
            row += f" & {self.total_weights[i]:.4f} & {self.ranks[i]}"
            out.append(row + r" \\")
        out += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(out)

    # ─── 可视化 ───
    def plot(self, save_path=None):
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                                'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("需要 matplotlib，请: pip install matplotlib")
            return
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].barh(self.criteria_names, self.w_criteria, color='steelblue')
        ax[0].set_xlabel('权重')
        ax[0].set_title(f'准则层权重（对 {self.goal_name}）')
        for i, v in enumerate(self.w_criteria):
            ax[0].text(v + 0.005, i, f'{v:.3f}', va='center')
        colors = ['gold' if i == self.ranking_idx[0] else 'lightcoral'
                  for i in range(len(self.alternative_names))]
        ax[1].barh(self.alternative_names, self.total_weights, color=colors)
        ax[1].set_xlabel('综合权重')
        ax[1].set_title('方案综合得分')
        for i, v in enumerate(self.total_weights):
            ax[1].text(v + 0.005, i, f'{v:.3f} (#{self.ranks[i]})', va='center')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


#
#                            交互式主流程
#
def interactive_main():
    print("\n" + "═" * 70)
    print("        层次分析法 (AHP) 交互式分析工具 · 数模竞赛版")
    print("═" * 70)

    # ① 基本信息
    print("\n【步骤 1/6】 设置基本信息")
    goal = input("  决策目标名称 [决策目标]: ").strip() or "决策目标"

    # ② 准则
    print("\n【步骤 2/6】 准则层设置")
    n = ask_int("  准则数量 n", default=3, min_val=2, max_val=15)
    criteria_names = []
    print(f"  请依次输入 {n} 个准则名称（回车跳过用默认名）：")
    for i in range(n):
        name = input(f"    准则 {i+1}: ").strip() or f"C{i+1}"
        criteria_names.append(name)

    # ③ 方案
    print("\n【步骤 3/6】 方案层设置")
    m = ask_int("  方案数量 m", default=3, min_val=2, max_val=15)
    alt_names = []
    print(f"  请依次输入 {m} 个方案名称（回车跳过用默认名）：")
    for i in range(m):
        name = input(f"    方案 {i+1}: ").strip() or f"A{i+1}"
        alt_names.append(name)

    # ④ 准则层判断矩阵
    print("\n【步骤 4/6】 输入准则层判断矩阵")
    criteria_matrix = input_matrix_interactive(
        n, criteria_names, f"准则层判断矩阵 ({n}×{n})")

    # ⑤ 方案层判断矩阵 (n 个)
    print("\n【步骤 5/6】 输入方案层判断矩阵")
    print(f"  需输入 {n} 个方案层判断矩阵（每个 {m}×{m}），每个对应一个准则")
    alt_matrices = []
    for i, cname in enumerate(criteria_names):
        am = input_matrix_interactive(
            m, alt_names, f"[{i+1}/{n}] 方案对「{cname}」的判断矩阵")
        alt_matrices.append(am)

    # ⑥ 计算方法
    print("\n【步骤 6/6】 计算与输出选项")
    print("  权重计算方法：")
    print("    1) 特征值法 (推荐，最精确)")
    print("    2) 算术平均法 (和法)")
    print("    3) 几何平均法 (根法)")
    method_map = {'1': 'eigen', '2': 'arithmetic', '3': 'geometric'}
    mc = input("  选择 [1]: ").strip() or "1"
    method = method_map.get(mc, 'eigen')

    # 计算
    print("\n" + "═" * 70)
    print("  正在计算...")
    print("═" * 70)
    try:
        model = AHPModel(criteria_matrix, alt_matrices,
                          criteria_names=criteria_names,
                          alternative_names=alt_names,
                          goal_name=goal, method=method)
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        return

    # 终端报告
    model.print_report()

    # 保存选项
    print("\n" + "═" * 70)
    print("  报告输出选项")
    print("═" * 70)

    if ask_yes_no("保存 Markdown 报告到 ahp_report.md？", 'y'):
        md = model.to_markdown()
        with open('ahp_report.md', 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"  ✓ 已保存到: {os.path.abspath('ahp_report.md')}")
        if ask_yes_no("  同时打印 Markdown 到终端 (方便复制)？", 'n'):
            print("\n" + "─" * 70)
            print(md)
            print("─" * 70)

    if ask_yes_no("保存 LaTeX 报告到 ahp_report.tex？", 'y'):
        lt = model.to_latex()
        with open('ahp_report.tex', 'w', encoding='utf-8') as f:
            f.write(lt)
        print(f"  ✓ 已保存到: {os.path.abspath('ahp_report.tex')}")
        if ask_yes_no("  同时打印 LaTeX 到终端 (方便复制)？", 'n'):
            print("\n" + "─" * 70)
            print(lt)
            print("─" * 70)

    if ask_yes_no("生成可视化图表 ahp_result.png？", 'n'):
        try:
            model.plot(save_path='ahp_result.png')
            print(f"  ✓ 已保存到: {os.path.abspath('ahp_result.png')}")
        except Exception as e:
            print(f"  ⚠ 可视化失败: {e}")

    print("\n" + "═" * 70)
    print(f"  ✨ 分析完成！最优方案: {model.ranking[0]}")
    print("═" * 70 + "\n")

    return model


# ============================================================================
#                                  入口
# ============================================================================
if __name__ == "__main__":
    try:
        interactive_main()
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
