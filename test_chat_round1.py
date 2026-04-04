#!/usr/bin/env python3
"""
BrainAI 自动化聊天测试脚本 (Round 1)

测试内容：
  - 第1轮：身份认知
  - 第2轮：记忆注入（姓名/城市/年龄/职业）
  - 第3轮：无关话题（干扰）
  - 第4轮：记忆召回（验证海马体是否记住用户信息）
  - 第5轮：推理能力（传递性推理）

评估维度：
  - 记忆测试：第4轮是否提到 "张三" 和 "程序员"
  - 推理测试：第5轮是否正确回答 "A 更高"
  - 异常检测：回复中是否包含乱码/异常字符
"""

import os
import sys
import time
import re
import traceback
from datetime import datetime
from typing import List, Dict


# ==================== 配置 ====================

WORK_DIR = "/home/z/my-project/stdpbrain"
OUTPUT_PATH = "/home/z/my-project/download/chat_test_round1.txt"
MODEL_PATH = "./models/Qwen3.5-0.8B"

MAX_TOKENS = 200          # 每轮最大生成 token 数
THINKING = False           # 关闭思考过程，加速测试
HISTORY_MAX_TURNS = 10     # 对话历史保留最近 N 条消息（N=5轮 x 2条）

# 5 轮测试对话
TEST_ROUNDS = [
    {"round": 1, "name": "身份认知",   "input": "你好，请介绍一下你自己。"},
    {"round": 2, "name": "记忆注入",   "input": "我叫张三，我来自北京，今年28岁，是一名程序员。"},
    {"round": 3, "name": "无关话题",   "input": "今天天气怎么样？"},
    {"round": 4, "name": "记忆召回",   "input": "你还记得我的名字和职业吗？"},
    {"round": 5, "name": "推理能力",   "input": "如果A比B高，B比C高，那A和C谁更高？"},
]


# ==================== 工具函数 ====================

def detect_garbled(text: str) -> List[str]:
    """
    检测文本中的乱码或异常字符。

    返回异常列表（空列表表示正常）。
    """
    issues = []

    if not text:
        issues.append("回复为空")
        return issues

    # 1. 连续重复字符 (3+ 个相同字符)
    for pattern in [r'(.)\1{4,}']:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(f"重复字符: {matches[:3]}")

    # 2. 不可见控制字符比例过高
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if len(text) > 0 and control_chars / len(text) > 0.1:
        issues.append(f"控制字符比例过高: {control_chars}/{len(text)}")

    # 3. 替换字符 (U+FFFD)
    if '\ufffd' in text:
        issues.append("包含替换字符 (U+FFFD)")

    # 4. 非常规 Unicode 区块（零宽字符等）
    zero_width = sum(1 for c in text if c in '\u200b\u200c\u200d\ufeff\u00ad')
    if zero_width > 0:
        issues.append(f"包含零宽字符: {zero_width}个")

    # 5. 回复中残留内部标签（模型泄漏系统 prompt）
    internal_labels = [
        "<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|",
        "[内心思维]", "[自闭环模式]", "[系统提示]", "【自闭环任务】",
        "(思考连续性)", "(当前感受)", "[意识状态：",
    ]
    found_labels = [label for label in internal_labels if label in text]
    if found_labels:
        issues.append(f"内部标签泄漏: {found_labels}")

    return issues


def evaluate_memory(round4_response: str) -> Dict:
    """
    评估记忆测试（第4轮）。

    通过条件：回复中同时包含 "张三" 和 "程序员"。
    """
    has_name = "张三" in round4_response
    has_job = "程序员" in round4_response

    passed = has_name and has_job

    details = {
        "passed": passed,
        "name_recalled": has_name,
        "job_recalled": has_job,
    }

    # 额外检测：是否提到了其他注入信息（加分项）
    extras = []
    if "北京" in round4_response:
        extras.append("城市")
    if "28" in round4_response:
        extras.append("年龄")
    details["bonus"] = extras

    return details


def evaluate_reasoning(round5_response: str) -> Dict:
    """
    评估推理测试（第5轮）。

    通过条件：回复中认为 A 更高（包含 "A更高" / "A比C高" / "A高" 等表述）。
    """
    text = round5_response.strip()

    # 正面模式：AI 认为 A 更高
    positive_patterns = [
        r'A\s*比\s*C\s*高',
        r'A\s*更\s*高',
        r'A\s*高',
        r'更高.*A',
        r'A.*更高',
    ]

    # 负面模式：AI 认为 C 更高 或 不确定
    negative_patterns = [
        r'C\s*比\s*A\s*高',
        r'C\s*更\s*高',
        r'C\s*高',
        r'无法判断',
        r'不确定',
        r'不一定',
        r'无法确定',
    ]

    positive_hits = sum(1 for p in positive_patterns if re.search(p, text))
    negative_hits = sum(1 for p in negative_patterns if re.search(p, text))

    passed = positive_hits > 0 and negative_hits == 0

    return {
        "passed": passed,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
    }


def format_report(
    results: List[Dict],
    memory_eval: Dict,
    reasoning_eval: Dict,
    total_time: float,
    exceptions: List[str],
) -> str:
    """生成评估报告文本。"""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("  BrainAI 自动化聊天测试报告 (Round 1)")
    lines.append(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  底座模型: Qwen3.5-0.8B  |  thinking=False  |  max_tokens={MAX_TOKENS}")
    lines.append(f"  总耗时: {total_time:.1f}s")
    lines.append(sep)

    # ---- 各轮对话详情 ----
    lines.append("")
    lines.append("[一] 各轮对话详情")
    lines.append("-" * 70)

    for r in results:
        lines.append(f"")
        lines.append(f"  第{r['round']}轮  [{r['name']}]")
        lines.append(f"  用户输入: {r['input']}")
        lines.append(f"  AI 回复:  {r['response']}")
        lines.append(f"  生成耗时: {r['elapsed']:.2f}s  |  回复长度: {len(r['response'])} 字")

        if r["garbled"]:
            lines.append(f"  ⚠ 异常检测: {', '.join(r['garbled'])}")
        else:
            lines.append(f"  ✓ 无乱码/异常")
        lines.append("")

    # ---- 记忆测试 ----
    lines.append("[二] 记忆测试 (第4轮)")
    lines.append("-" * 70)
    lines.append(f"  结果: {'✓ 通过' if memory_eval['passed'] else '✗ 未通过'}")
    lines.append(f"  - 姓名召回 ('张三'): {'✓' if memory_eval['name_recalled'] else '✗'}")
    lines.append(f"  - 职业召回 ('程序员'): {'✓' if memory_eval['job_recalled'] else '✗'}")
    if memory_eval.get("bonus"):
        lines.append(f"  - 额外信息: {', '.join(memory_eval['bonus'])}")
    lines.append("")

    # ---- 推理测试 ----
    lines.append("[三] 推理测试 (第5轮)")
    lines.append("-" * 70)
    lines.append(f"  结果: {'✓ 通过' if reasoning_eval['passed'] else '✗ 未通过'}")
    lines.append(f"  - 正面匹配数 (A更高): {reasoning_eval['positive_hits']}")
    lines.append(f"  - 负面匹配数 (C更高/不确定): {reasoning_eval['negative_hits']}")
    lines.append("")

    # ---- 汇总 ----
    lines.append("[四] 汇总")
    lines.append("-" * 70)
    mem_status = "✓ 通过" if memory_eval["passed"] else "✗ 未通过"
    rea_status = "✓ 通过" if reasoning_eval["passed"] else "✗ 未通过"
    total_passed = sum(1 for r in results if not r["garbled"])
    total_rounds = len(results)
    lines.append(f"  记忆测试: {mem_status}")
    lines.append(f"  推理测试: {rea_status}")
    lines.append(f"  乱码检测: {total_passed}/{total_rounds} 轮无异常")

    if exceptions:
        lines.append(f"  异常记录: {len(exceptions)} 条")
        for ex in exceptions:
            lines.append(f"    - {ex}")
    else:
        lines.append(f"  异常记录: 无")

    # 总评
    all_passed = memory_eval["passed"] and reasoning_eval["passed"] and not exceptions
    overall = "★★★ 全部通过 ★★★" if all_passed else "★ 部分未通过 ★"
    lines.append("")
    lines.append(f"  综合评定: {overall}")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ==================== 主流程 ====================

def main():
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=" * 70)
    print("  BrainAI 自动化聊天测试 (Round 1)")
    print("  底座模型: Qwen3.5-0.8B  |  thinking=False")
    print(f"  输出: {OUTPUT_PATH}")
    print("=" * 70)

    # ---- 1. 加载系统 ----
    print("\n[初始化] 正在加载 BrainAI 系统...")
    try:
        from configs.arch_config import BrainAIConfig
        from core.interfaces import BrainAIInterface

        config = BrainAIConfig()
        config.model_path = MODEL_PATH

        # 量化配置
        import config as user_config
        config.quantization = getattr(user_config, 'QUANTIZATION', 'FP16')
        config.QUANTIZATION = config.quantization

        load_start = time.time()
        ai = BrainAIInterface(config)
        load_time = time.time() - load_start
        print(f"[初始化] 加载完成，耗时 {load_time:.1f}s")
    except Exception as e:
        msg = f"系统加载失败: {e}"
        print(f"[错误] {msg}")
        traceback.print_exc()
        # 写入错误报告
        report = f"BrainAI 测试报告\n{'='*50}\n错误: {msg}\n{traceback.format_exc()}"
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        return

    # ---- 2. 执行 5 轮对话 ----
    results = []
    exceptions = []
    history: List[Dict[str, str]] = []
    total_start = time.time()

    for round_info in TEST_ROUNDS:
        round_num = round_info["round"]
        round_name = round_info["name"]
        user_input = round_info["input"]

        print(f"\n{'='*70}")
        print(f"  第{round_num}轮  [{round_name}]")
        print(f"{'='*70}")
        print(f"  用户: {user_input}")

        try:
            gen_start = time.time()
            response = ai.chat(
                user_input,
                history=history,
                max_tokens=MAX_TOKENS,
                thinking=THINKING,
            )
            elapsed = time.time() - gen_start

            # 清理空响应
            if response is None:
                response = ""
            response = str(response).strip()

            # 检测乱码
            garbled = detect_garbled(response)

            print(f"  AI:   {response[:500]}{'...' if len(response) > 500 else ''}")
            print(f"  耗时: {elapsed:.2f}s  |  长度: {len(response)}字")
            if garbled:
                print(f"  ⚠ 异常: {', '.join(garbled)}")

            results.append({
                "round": round_num,
                "name": round_name,
                "input": user_input,
                "response": response,
                "elapsed": elapsed,
                "garbled": garbled,
                "error": None,
            })

            # 更新对话历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            # 限制历史长度，避免 prompt 过长
            if len(history) > HISTORY_MAX_TURNS * 2:
                history = history[-(HISTORY_MAX_TURNS * 2):]

        except Exception as e:
            error_msg = f"第{round_num}轮异常: {type(e).__name__}: {e}"
            print(f"  ✗ 错误: {error_msg}")
            traceback.print_exc()
            exceptions.append(error_msg)
            results.append({
                "round": round_num,
                "name": round_name,
                "input": user_input,
                "response": "",
                "elapsed": 0,
                "garbled": ["异常终止"],
                "error": error_msg,
            })

    total_time = time.time() - total_start

    # ---- 3. 评估 ----
    # 取第4轮结果做记忆评估
    round4_response = ""
    for r in results:
        if r["round"] == 4:
            round4_response = r["response"]
            break
    memory_eval = evaluate_memory(round4_response)

    # 取第5轮结果做推理评估
    round5_response = ""
    for r in results:
        if r["round"] == 5:
            round5_response = r["response"]
            break
    reasoning_eval = evaluate_reasoning(round5_response)

    # ---- 4. 生成并保存报告 ----
    report = format_report(results, memory_eval, reasoning_eval, total_time, exceptions)

    print("\n" + report)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[保存] 报告已写入: {OUTPUT_PATH}")

    # ---- 5. 可选：打印系统统计 ----
    try:
        stats = ai.get_stats()
        print(f"\n[系统统计]")
        print(f"  海马体记忆数: {stats['hippocampus']['num_memories']}")
        print(f"  STDP 周期数: {stats['stdp']['cycle_count']}")
        if ai.inner_thought_engine:
            print(f"  思维状态: {ai.inner_thought_engine.mind_state.value}")
    except Exception:
        pass

    print("\n测试完成。")


if __name__ == "__main__":
    main()
