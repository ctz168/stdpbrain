#!/usr/bin/env python3
"""
类人脑双系统全闭环 AI架构 - 主入口

基于Qwen3.5-0.8B 底座模型
实现海马体 - 新皮层双系统类人脑架构

使用示例:
    python main.py --mode chat
    python main.py --mode generate --input "你好"
    python main.py --mode eval
"""

import argparse
import sys
import time
from typing import Optional

# 导入配置
import config as secret_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="类人脑双系统全闭环 AI架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对话模式
  python main.py --mode chat
  
  # 生成模式
  python main.py --mode generate --input "请解释量子力学"
  
  # 评测模式
  python main.py --mode eval
  
  # 查看统计
  python main.py --mode stats
        """
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="chat",
        choices=["chat", "generate", "eval", "stats", "telegram", "evaluate"],
        help="运行模式"
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default="",
        help="输入文本 (generate 模式)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=secret_config.MODEL_PATH,
        help="Qwen3.5-0.8B 模型路径"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=secret_config.DEVICE or None,
        choices=["cuda", "cpu", "mps", None],
        help="运行设备"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        default=secret_config.QUANTIZATION,
        choices=["FP16", "INT8", "INT4"],
        help="量化类型"
    )
    
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=secret_config.TELEGRAM_BOT_TOKEN,
        help="Telegram Bot Token (telegram 模式)"
    )
    
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="启用异步模式 (telegram 模式)"
    )
    
    return parser.parse_args()


def run_chat(ai):
    """对话模式"""
    print("=" * 60)
    print("类人脑AI - 对话模式")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    history = []
    
    while True:
        try:
            user_input = input("\n你：").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input:
                continue
            
            # 生成回复
            start_time = time.time()
            response = ai.chat(user_input, history)
            elapsed = time.time() - start_time
            
            print(f"\nAI: {response}")
            print(f"[耗时：{elapsed*1000:.1f}ms]")
            
            # 更新历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[错误] {e}")


def run_generate(ai, input_text: str):
    """生成模式"""
    print("=" * 60)
    print("类人脑AI - 生成模式")
    print("=" * 60)
    
    if not input_text:
        print("[错误] 请提供输入文本 (--input)")
        return
    
    print(f"\n输入：{input_text}")
    print("\n生成中...")
    
    start_time = time.time()
    output = ai.generate(input_text, max_tokens=200)
    elapsed = time.time() - start_time
    
    print(f"\n输出:\n{output.text}")
    print(f"\n[统计]")
    print(f"  生成 token 数：{len(output.tokens)}")
    print(f"  置信度：{output.confidence:.2%}")
    print(f"  耗时：{elapsed*1000:.1f}ms")
    print(f"  记忆锚点数：{len(output.memory_anchors)}")


def run_evaluation(ai):
    """评测模式"""
    print("=" * 60)
    print("类人脑AI - 综合评测")
    print("=" * 60)
    
    from evaluation.evaluator import BrainAIEvaluator
    
    evaluator = BrainAIEvaluator(ai)
    
    # 执行评测
    results = evaluator.run_all_evaluations()
    
    # 打印结果
    print("\n[评测结果]")
    print(f"  海马体记忆能力：{results['hippocampus_score']:.2f}/1.0")
    print(f"  基础能力对标：{results['base_capability_score']:.2f}/1.0")
    print(f"  逻辑推理能力：{results['reasoning_score']:.2f}/1.0")
    print(f"  端侧性能：{results['edge_performance_score']:.2f}/1.0")
    print(f"  自闭环优化：{results['self_loop_score']:.2f}/1.0")
    print(f"\n  总分：{results['total_score']:.2f}/1.0")


def run_stats(ai):
    """统计模式"""
    print("=" * 60)
    print("类人脑AI - 系统统计")
    print("=" * 60)
    
    stats = ai.get_stats()
    
    print("\n[海马体系统]")
    print(f"  记忆数量：{stats['hippocampus']['num_memories']}")
    print(f"  内存使用：{stats['hippocampus']['memory_usage_mb']:.2f}MB")
    print(f"  最大内存：{stats['hippocampus']['max_memory_mb']:.2f}MB")
    
    print("\n[STDP 引擎]")
    print(f"  周期计数：{stats['stdp']['cycle_count']}")
    print(f"  追踪激活数：{stats['stdp']['num_tracked_activations']}")
    
    print("\n[100Hz 推理引擎]")
    print(f"  总周期数：{stats['refresh_engine']['total_cycles']}")
    print(f"  平均周期时间：{stats['refresh_engine']['avg_cycle_time_ms']:.2f}ms")
    print(f"  最大周期时间：{stats['refresh_engine']['max_cycle_time_ms']:.2f}ms")
    print(f"  超时次数：{stats['refresh_engine']['overrun_count']}")
    
    print("\n[自闭环优化]")
    print(f"  周期计数：{stats['self_loop']['cycle_count']}")
    print(f"  平均准确率：{stats['self_loop']['avg_accuracy']:.2%}")


def run_telegram_bot(ai, token: str = None, async_mode: bool = False):
    """Telegram Bot 模式"""
    print("=" * 60)
    print("类人脑AI - Telegram Bot")
    print("=" * 60)
    
    from telegram_bot.bot import BrainAIBot
    
    # 获取 Token
    bot_token = token or secret_config.TELEGRAM_BOT_TOKEN
    
    # 创建 Bot
    bot = BrainAIBot(
        token=bot_token,
        ai_interface=ai,
        stream_chunk_size=1,
        stream_delay_ms=50,
        proxy_url=secret_config.PROXY_URL
    )
    
    print(f"\n[Bot] Token: {bot_token[:20]}...")
    print("[Bot] 启动中...")
    print("\n按 Ctrl+C 停止 Bot")
    print("=" * 60)
    
    try:
        if async_mode:
            import asyncio
            asyncio.run(bot.start_async())
        else:
            bot.run()
    except KeyboardInterrupt:
        print("\n[Bot] 已停止")
    except Exception as e:
        print(f"[错误] Bot 运行失败：{e}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("类人脑双系统全闭环 AI架构")
    print("底座模型：Qwen3.5-0.8B")
    print("=" * 60)
    
    # ========== 1. 初始化配置 ==========
    from configs.arch_config import BrainAIConfig
    
    config = BrainAIConfig()
    config.model_path = args.model_path
    
    # ========== 2. 创建 AI 实例 ==========
    print("\n[初始化] 加载模型和模块...")
    
    try:
        from core.interfaces import BrainAIInterface
        
        ai = BrainAIInterface(config, device=args.device)
        print("[初始化] 完成 ✓")
        
    except Exception as e:
        print(f"[错误] 初始化失败：{e}")
        print("\n提示：请确保已下载 Qwen3.5-0.8B 模型到指定路径")
        print("可使用以下命令下载:")
        print("  huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ./models/Qwen3.5-0.8B")
        sys.exit(1)
    
    # ========== 3. 执行对应模式 ==========
    try:
        if args.mode == "chat":
            run_chat(ai)
        
        elif args.mode == "generate":
            run_generate(ai, args.input)
        
        elif args.mode == "eval":
            run_evaluation(ai)
        
        elif args.mode == "stats":
            run_stats(ai)
        
        elif args.mode == "telegram":
            run_telegram_bot(ai, args.telegram_token, args.async_mode)
        
        elif args.mode == "evaluate":
            run_automated_evaluation(ai)
    
    finally:
        # ========== 4. 清理 (睡眠固化) ==========
        print("\n[退出] 正在固化记忆与意识状态...")
        try:
            ai.save_state("brain_state.pt")
        except Exception as e:
            print(f"[警告] 状态保存失败: {e}")
    
    print("\n再见！")


def run_automated_evaluation(ai):
    """自动化评估模式"""
    print("=" * 60)
    print("类人脑AI - 自动化能力评估")
    print("=" * 60)

    report = {}
    history = []

    def chat_and_record(user_input, step_name):
        print(f"\n--- {step_name} ---")
        print(f"你：{user_input}")
        response = ai.chat(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        print(f"AI: {response}")
        return response

    # 1. 身份认知
    report['identity_check'] = chat_and_record("你好，介绍一下你自己。", "Step 1: 身份认知测试")

    # 2. 获取初始状态
    stats_before = ai.get_stats()
    report['stats_before'] = stats_before
    print("\n--- Step 2: 记录初始状态 ---")
    print(f"海马体记忆数: {stats_before['hippocampus']['num_memories']}")
    print(f"STDP 动态权重范数: {stats_before['stdp']['dynamic_weight_norm']:.6f}")

    # 3. 注入新记忆
    chat_and_record("我叫张三，我来自北京。", "Step 3: 注入新记忆")

    # 4. 无关对话
    chat_and_record("今天天气怎么样？", "Step 4: 无关对话")

    # 5. 记忆调用测试
    report['memory_recall'] = chat_and_record("你还记得我叫什么名字吗？", "Step 5: 记忆调用测试")

    # 6. 获取最终状态
    stats_after = ai.get_stats()
    report['stats_after'] = stats_after
    print("\n--- Step 6: 记录最终状态 ---")
    print(f"海马体记忆数: {stats_after['hippocampus']['num_memories']}")
    print(f"STDP 动态权重范数: {stats_after['stdp']['dynamic_weight_norm']:.6f}")

    # 7. 生成评估报告
    print("\n" + "=" * 60)
    print("自动化评估报告")
    print("=" * 60)
    
    # 回复质量
    print("\n[1. 回复质量]")
    print(f"  - 身份认知回复: {report['identity_check']}")
    print(f"  - 记忆调用回复: {report['memory_recall']}")

    # 学习能力
    mem_diff = stats_after['hippocampus']['num_memories'] - stats_before['hippocampus']['num_memories']
    stdp_diff = stats_after['stdp']['dynamic_weight_norm'] - stats_before['stdp']['dynamic_weight_norm']
    print("\n[2. 学习能力]")
    print(f"  - 海马体记忆增长: {mem_diff} (期望 > 0)")
    print(f"  - STDP 权重变化: {stdp_diff:.6f} (期望 != 0)")

    # 智力与自我意识
    print("\n[3. 智力与自我意识]")
    identity_ok = "AI" in report['identity_check'] or "模型" in report['identity_check']
    recall_ok = "张三" in report['memory_recall']
    print(f"  - 自我意识一致性: {'通过' if identity_ok else '失败'}")
    print(f"  - 短期记忆准确性: {'通过' if recall_ok else '失败'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
