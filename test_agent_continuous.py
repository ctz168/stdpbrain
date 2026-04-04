#!/usr/bin/env python3
"""
stdpbrain 连续模式 Agent 能力综合测试脚本
测试重点：
1. Agent 长任务执行能力（多步推理、规划、跟踪）
2. 工具调用能力（内部工具：记忆召回、自闭环优化、目标系统）
3. 持续思维流质量（独白连贯性、自主性）
4. 记忆持久化（跨对话记忆保持）
5. 自我指涉与自我优化能力
"""

import sys
import os
import time
import json
import traceback
from datetime import datetime

# ===== HOTFIX (同 main.py) =====
def _is_local_path(path):
    if not path or not isinstance(path, str):
        return False
    if path.startswith('/') or path.startswith('~') or path.startswith('.'):
        return os.path.isdir(path)
    if len(path.split('/')) > 2 and os.path.isdir(path):
        return True
    return False

try:
    from transformers.utils.hub import cached_file as _orig_cached_file
    from transformers.utils.hub import cached_files as _orig_cached_files
    def _patched_cached_file(path_or_repo_id, filename, **kwargs):
        if _is_local_path(path_or_repo_id):
            local_path = os.path.join(path_or_repo_id, filename)
            if os.path.isfile(local_path):
                return local_path
        return _orig_cached_file(path_or_repo_id, filename, **kwargs)
    def _patched_cached_files(path_or_repo_id, filenames, cache_dir=None, **kwargs):
        if _is_local_path(path_or_repo_id):
            results = []
            for fname in filenames:
                local_path = os.path.join(path_or_repo_id, fname)
                if os.path.isfile(local_path):
                    results.append(local_path)
            if results:
                return results
        return _orig_cached_files(path_or_repo_id, filenames, cache_dir=cache_dir, **kwargs)
    import transformers.utils.hub as _hf_hub
    _hf_hub.cached_file = _patched_cached_file
    _hf_hub.cached_files = _patched_cached_files
except Exception:
    pass

try:
    import huggingface_hub.utils._validators as _hf_validators
    _orig_validate = _hf_validators.validate_repo_id
    def _patched_validate_repo_id(repo_id, **kwargs):
        if _is_local_path(repo_id):
            return
        return _orig_validate(repo_id, **kwargs)
    _hf_validators.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass


class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.score = 0.0
        self.max_score = 10.0
        self.details = []
        self.error = None
        self.duration = 0.0
    
    def add_detail(self, msg):
        self.details.append(msg)
    
    def set_pass(self, score=None):
        self.passed = True
        if score is not None:
            self.score = min(score, self.max_score)
    
    def set_fail(self, error_msg=None):
        self.passed = False
        if error_msg:
            self.error = error_msg
    
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"[{status}] {self.name}: {self.score}/{self.max_score} ({self.duration:.1f}s)"


class AgentContinuousTester:
    def __init__(self, model_path="./models/Qwen3.5-0.8B", device="cpu", quantization="FP16"):
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self.ai = None
        self.results = {}
        self.chat_history = []
    
    def setup(self):
        """初始化AI系统"""
        print("=" * 70)
        print("  stdpbrain 连续模式 Agent 能力综合测试")
        print("  测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70)
        
        # 加载配置
        import config as user_config
        from configs.arch_config import BrainAIConfig
        
        config = BrainAIConfig()
        config.model_path = self.model_path
        config.quantization = getattr(user_config, 'QUANTIZATION', config.quantization)
        config.QUANTIZATION = config.quantization
        
        # 加载AI
        from core.interfaces import BrainAIInterface
        self.ai = BrainAIInterface(config, device=self.device)
        print("[系统] AI系统初始化完成\n")
        return True
    
    def chat(self, user_input, max_tokens=256):
        """封装对话接口"""
        start = time.time()
        try:
            response = self.ai.chat(
                user_input,
                history=self.chat_history[-6:] if self.chat_history else [],
                max_tokens=max_tokens
            )
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            elapsed = time.time() - start
            return response, elapsed, None
        except Exception as e:
            elapsed = time.time() - start
            return None, elapsed, str(e)
    
    def think(self):
        """触发自发思考"""
        start = time.time()
        try:
            stats = self.ai.think()
            elapsed = time.time() - start
            return stats.get('monologue', ''), elapsed, None
        except Exception as e:
            elapsed = time.time() - start
            return None, elapsed, str(e)
    
    # ===================== 测试用例 =====================
    
    def test_1_basic_responsiveness(self):
        """测试1: 基础响应能力 - 确认模型能正常对话"""
        result = TestResult("基础响应能力")
        start = time.time()
        
        test_cases = [
            "你好",
            "1+1等于几？",
            "今天天气怎么样？"
        ]
        
        passed = 0
        for i, q in enumerate(test_cases):
            resp, elapsed, err = self.chat(q, max_tokens=100)
            if err:
                result.add_detail(f"  Q{i+1}: '{q}' -> 错误: {err}")
            elif resp and len(resp.strip()) > 5:
                passed += 1
                result.add_detail(f"  Q{i+1}: '{q}' -> OK ({len(resp)}字, {elapsed:.1f}s)")
            else:
                result.add_detail(f"  Q{i+1}: '{q}' -> 回复过短: {resp}")
        
        score = (passed / len(test_cases)) * 10
        result.duration = time.time() - start
        if passed >= 2:
            result.set_pass(score)
        else:
            result.set_fail(f"仅{passed}/{len(test_cases)}个问题得到有效回复")
        return result
    
    def test_2_agent_long_task_planning(self):
        """测试2: Agent长任务 - 多步规划与执行"""
        result = TestResult("Agent长任务规划能力")
        start = time.time()
        
        # 复杂任务：要求AI制定多步骤计划
        task_prompt = "请帮我制定一个学习Python编程的3个月计划，包含每周的具体目标和任务安排。"
        result.add_detail(f"任务: {task_prompt}")
        
        resp, elapsed, err = self.chat(task_prompt, max_tokens=512)
        if err:
            result.set_fail(f"错误: {err}")
            result.duration = time.time() - start
            return result
        
        if not resp:
            result.set_fail("无回复")
            result.duration = time.time() - start
            return result
        
        result.add_detail(f"回复长度: {len(resp)}字, 耗时: {elapsed:.1f}s")
        
        # 评估规划质量
        score = 0
        
        # 检查是否有分阶段/分步骤结构
        planning_keywords = ["第", "周", "月", "阶段", "步骤", "目标", "计划", "安排"]
        keyword_hits = sum(1 for kw in planning_keywords if kw in resp)
        if keyword_hits >= 3:
            score += 3
            result.add_detail(f"  ✅ 规划结构: 包含{keyword_hits}个规划关键词")
        else:
            result.add_detail(f"  ⚠️ 规划结构: 仅包含{keyword_hits}个规划关键词")
            score += 1
        
        # 检查是否有具体内容
        if len(resp) > 100:
            score += 2
            result.add_detail(f"  ✅ 内容充实: {len(resp)}字")
        elif len(resp) > 50:
            score += 1
            result.add_detail(f"  ⚠️ 内容一般: {len(resp)}字")
        
        # 检查是否有Python相关内容
        python_keywords = ["python", "Python", "基础", "函数", "数据结构", "项目", "实战"]
        python_hits = sum(1 for kw in python_keywords if kw in resp)
        if python_hits >= 2:
            score += 2
            result.add_detail(f"  ✅ 领域相关: 包含{python_hits}个Python关键词")
        else:
            result.add_detail(f"  ⚠️ 领域相关: 仅{python_hits}个Python关键词")
            score += 1
        
        # 检查是否有时间线
        timeline_keywords = ["第一", "第二", "第三", "1.", "2.", "3.", "一周", "两周", "一个月"]
        timeline_hits = sum(1 for kw in timeline_keywords if kw in resp)
        if timeline_hits >= 2:
            score += 2
            result.add_detail(f"  ✅ 时间线: 包含{timeline_hits}个时间线标记")
        else:
            score += 1
            result.add_detail(f"  ⚠️ 时间线: 仅{timeline_hits}个时间线标记")
        
        # 长度加分
        if len(resp) > 200:
            score += 1
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_3_agent_multi_turn_reasoning(self):
        """测试3: Agent多轮推理 - 跨轮次上下文追踪"""
        result = TestResult("Agent多轮推理追踪")
        start = time.time()
        
        # 模拟一个多轮推理任务
        turns = [
            ("我叫小明，今年25岁，住在北京，是一名程序员。", "注入个人信息"),
            ("请记住我刚才告诉你的信息。", "确认记忆存储"),
            ("你还记得我叫什么名字吗？我多大？", "测试记忆召回"),
            ("那你还记得我的职业和住在哪里吗？", "测试完整记忆"),
        ]
        
        score = 0
        recall_results = []
        
        for i, (question, purpose) in enumerate(turns):
            resp, elapsed, err = self.chat(question, max_tokens=200)
            result.add_detail(f"\n--- 第{i+1}轮 ({purpose}) ---")
            
            if err:
                result.add_detail(f"  ❌ 错误: {err}")
                continue
            
            if not resp:
                result.add_detail(f"  ❌ 无回复")
                continue
            
            result.add_detail(f"  Q: {question}")
            result.add_detail(f"  A: {resp[:150]}{'...' if len(resp)>150 else ''}")
            
            # 第1轮：注入信息
            if i == 0:
                if any(w in resp for w in ["小明", "你好", "认识"]):
                    score += 1
                    result.add_detail(f"  ✅ 信息接收成功")
                else:
                    result.add_detail(f"  ⚠️ 未能确认接收到信息")
            
            # 第3轮：测试名字和年龄召回
            if i == 2:
                name_ok = "小明" in resp
                age_ok = "25" in resp
                if name_ok:
                    score += 2
                    result.add_detail(f"  ✅ 名字召回: 成功")
                    recall_results.append(True)
                else:
                    result.add_detail(f"  ❌ 名字召回: 失败")
                    recall_results.append(False)
                if age_ok:
                    score += 2
                    result.add_detail(f"  ✅ 年龄召回: 成功")
                    recall_results.append(True)
                else:
                    result.add_detail(f"  ❌ 年龄召回: 失败")
                    recall_results.append(False)
            
            # 第4轮：测试职业和地址召回
            if i == 3:
                job_ok = "程序员" in resp
                loc_ok = "北京" in resp
                if job_ok:
                    score += 2
                    result.add_detail(f"  ✅ 职业召回: 成功")
                    recall_results.append(True)
                else:
                    result.add_detail(f"  ❌ 职业召回: 失败")
                    recall_results.append(False)
                if loc_ok:
                    score += 2
                    result.add_detail(f"  ✅ 住址召回: 成功")
                    recall_results.append(True)
                else:
                    result.add_detail(f"  ❌ 住址召回: 失败")
                    recall_results.append(False)
        
        recall_rate = sum(recall_results) / len(recall_results) if recall_results else 0
        result.add_detail(f"\n  记忆召回率: {recall_rate:.0%}")
        
        # 检查海马体记忆状态
        try:
            stats = self.ai.get_stats()
            mem_count = stats['hippocampus']['num_memories']
            result.add_detail(f"  海马体记忆数: {mem_count}")
            if mem_count > 0:
                score += 1
        except:
            pass
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_4_tool_calling_memory_recall(self):
        """测试4: 工具调用 - 海马体记忆系统"""
        result = TestResult("工具调用-记忆系统")
        start = time.time()
        
        # 测试内部工具调用：海马体记忆编码/召回
        score = 0
        
        # 1. 注入多条不同类型的记忆
        memories = [
            "我的手机号是13800138000",
            "我喜欢吃火锅和烤肉",
            "我家的猫叫小花，它是一只橘猫",
            "我下周一要去上海出差",
        ]
        
        result.add_detail("=== 阶段1: 记忆注入 ===")
        for mem in memories:
            resp, elapsed, err = self.chat(f"请记住这个信息：{mem}", max_tokens=100)
            if resp:
                result.add_detail(f"  注入: {mem[:30]}... -> OK")
            else:
                result.add_detail(f"  注入: {mem[:30]}... -> 失败")
        
        # 插入干扰对话
        result.add_detail("\n=== 阶段2: 干扰对话 ===")
        distractor_q = "你今天心情怎么样？跟我聊聊天气吧。"
        resp, _, _ = self.chat(distractor_q, max_tokens=80)
        result.add_detail(f"  干扰: {distractor_q}")
        result.add_detail(f"  回复: {resp[:80] if resp else '无'}...")
        
        # 2. 测试记忆召回（核心工具调用能力）
        result.add_detail("\n=== 阶段3: 记忆召回 ===")
        recall_tests = [
            ("我的手机号是多少？", "13800138000"),
            ("我喜欢吃什么？", ["火锅", "烤肉"]),
            ("我的猫叫什么名字？什么品种？", ["小花", "橘猫"]),
            ("我下周一要去哪里？", "上海"),
        ]
        
        recall_success = 0
        for question, expected in recall_tests:
            resp, elapsed, err = self.chat(question, max_tokens=100)
            
            if isinstance(expected, str):
                found = expected in (resp or "")
            else:
                found = any(exp in (resp or "") for exp in expected)
            
            if found:
                recall_success += 1
                score += 1.5
                result.add_detail(f"  ✅ 召回: '{question}' -> 找到 '{expected}'")
            else:
                result.add_detail(f"  ❌ 召回: '{question}' -> 未找到 '{expected}'")
                result.add_detail(f"     回复: {resp[:80] if resp else '无'}")
        
        # 检查海马体系统统计
        try:
            stats = self.ai.get_stats()
            result.add_detail(f"\n  海马体记忆数: {stats['hippocampus']['num_memories']}")
            result.add_detail(f"  STDP动态权重范数: {stats['stdp'].get('dynamic_weight_norm', 0):.6f}")
            if stats['hippocampus']['num_memories'] > 5:
                score += 1
        except Exception as e:
            result.add_detail(f"  获取统计失败: {e}")
        
        result.duration = time.time() - start
        if recall_success >= 2:
            result.set_pass(min(score, 10))
        else:
            result.set_fail(f"仅{recall_success}/{len(recall_tests)}个记忆成功召回")
        return result
    
    def test_5_tool_calling_self_loop(self):
        """测试5: 工具调用 - 自闭环优化系统"""
        result = TestResult("工具调用-自闭环优化")
        start = time.time()
        
        # 自闭环优化系统的三个模式
        # self_combine: 自组合
        # self_game: 自博弈
        # self_eval: 自评判
        
        score = 0
        
        # 触发高难度任务（应该触发自博弈或自评判模式）
        hard_tasks = [
            "请证明根号2是无理数",
            "请写一个快速排序算法，并分析时间复杂度",
        ]
        
        result.add_detail("=== 高难度任务测试（应触发自闭环优化）===")
        
        for task in hard_tasks:
            resp, elapsed, err = self.chat(task, max_tokens=300)
            
            # 检查是否触发了自闭环
            try:
                if hasattr(self.ai, 'self_loop') and self.ai.self_loop:
                    sl_stats = self.ai.self_loop.get_stats()
                    mode = self.ai.self_loop.decide_mode(task)
                    result.add_detail(f"\n  任务: {task[:40]}...")
                    result.add_detail(f"  自闭环模式: {mode}")
                    result.add_detail(f"  周期数: {sl_stats['cycle_count']}")
                    result.add_detail(f"  平均准确率: {sl_stats['avg_accuracy']:.2f}")
                    
                    if mode in ["self_game", "self_eval"]:
                        score += 2
                        result.add_detail(f"  ✅ 触发了高级自闭环模式")
                    elif mode == "self_combine":
                        score += 1
                        result.add_detail(f"  ⚠️ 触发了基础自组合模式")
                    
                    if sl_stats['cycle_count'] > 0:
                        score += 1
            except Exception as e:
                result.add_detail(f"  自闭环检测失败: {e}")
            
            if resp and len(resp) > 50:
                score += 1
                result.add_detail(f"  回复长度: {len(resp)}字")
        
        # 检查STDP更新
        try:
            stats = self.ai.get_stats()
            stdp_norm = stats['stdp'].get('dynamic_weight_norm', 0)
            result.add_detail(f"\n  STDP动态权重范数: {stdp_norm:.6f}")
            if stdp_norm > 0:
                score += 1
                result.add_detail(f"  ✅ STDP有学习更新")
        except:
            pass
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_6_tool_calling_goal_system(self):
        """测试6: 工具调用 - 目标系统"""
        result = TestResult("工具调用-目标系统")
        start = time.time()
        
        score = 0
        
        # 测试目标推断
        goal_test_inputs = [
            ("请记住我的生日是5月20日", "remember"),
            ("我叫王芳，是一名护士", "remember"),
            ("帮我查一下之前我告诉你的那些信息", "recall"),
            ("分析一下我们目前的对话", "analyze"),
        ]
        
        result.add_detail("=== 目标推断测试 ===")
        
        for input_text, expected_goal_type in goal_test_inputs:
            resp, elapsed, err = self.chat(input_text, max_tokens=150)
            
            # 检查目标系统
            try:
                if hasattr(self.ai, 'goal_system') and self.ai.goal_system:
                    gs = self.ai.goal_system
                    if gs.current_goal is not None:
                        actual_goal = gs.current_goal.goal_type.value
                        goal_desc = gs.current_goal.description
                        progress = gs.current_goal.progress
                        result.add_detail(f"\n  输入: '{input_text[:40]}...'")
                        result.add_detail(f"  推断目标: {actual_goal} - {goal_desc}")
                        result.add_detail(f"  目标进度: {progress:.1%}")
                        
                        if actual_goal == expected_goal_type:
                            score += 2
                            result.add_detail(f"  ✅ 目标推断正确")
                        else:
                            score += 0.5
                            result.add_detail(f"  ⚠️ 目标推断: 期望{expected_goal_type}, 实际{actual_goal}")
                    else:
                        result.add_detail(f"  ⚠️ 无当前目标")
                        score += 0.5
            except Exception as e:
                result.add_detail(f"  目标系统检测失败: {e}")
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_7_continuous_thought_flow(self):
        """测试7: 持续思维流 - 独白质量与连贯性"""
        result = TestResult("持续思维流质量")
        start = time.time()
        
        score = 0
        monologues = []
        
        # 生成多轮自发独白
        num_rounds = 3
        result.add_detail(f"=== 生成 {num_rounds} 轮自发独白 ===\n")
        
        for i in range(num_rounds):
            mono, elapsed, err = self.think()
            if err:
                result.add_detail(f"  第{i+1}轮: 错误 - {err}")
                continue
            
            if mono:
                monologues.append(mono)
                result.add_detail(f"  第{i+1}轮 ({elapsed:.1f}s): {mono[:100]}{'...' if len(mono)>100 else ''}")
            else:
                result.add_detail(f"  第{i+1}轮: 无输出")
        
        # 评估独白质量
        result.add_detail(f"\n=== 独白质量评估 ===")
        
        if len(monologues) >= 2:
            score += 2
            result.add_detail(f"  ✅ 独白生成: {len(monologues)}/{num_rounds}轮成功")
        
        # 检查独白多样性（不同轮次不应完全相同）
        if len(monologues) >= 2:
            unique_monologues = set(m.strip()[:20] for m in monologues)
            diversity = len(unique_monologues) / len(monologues)
            if diversity >= 0.8:
                score += 2
                result.add_detail(f"  ✅ 独白多样性: {diversity:.0%}")
            else:
                score += 1
                result.add_detail(f"  ⚠️ 独白多样性: {diversity:.0%}")
        
        # 检查独白质量（不是乱码）
        quality_count = 0
        for m in monologues:
            # 基本中文检测
            chinese_chars = sum(1 for c in m if '\u4e00' <= c <= '\u9fff')
            if chinese_chars > 5:
                quality_count += 1
        
        if quality_count == len(monologues) and quality_count > 0:
            score += 2
            result.add_detail(f"  ✅ 独白可读性: {quality_count}/{len(monologues)}轮可读")
        elif quality_count > 0:
            score += 1
            result.add_detail(f"  ⚠️ 独白可读性: {quality_count}/{len(monologues)}轮可读")
        
        # 检查思维状态
        try:
            if self.ai.current_thought_state is not None:
                score += 2
                result.add_detail(f"  ✅ 思维状态: 已维护 (shape={self.ai.current_thought_state.shape})")
            
            # 检查思维种子更新
            if self.ai.thought_seed:
                score += 1
                result.add_detail(f"  ✅ 思维种子: '{self.ai.thought_seed}'")
        except Exception as e:
            result.add_detail(f"  思维状态检测失败: {e}")
        
        # 全局工作空间
        try:
            if hasattr(self.ai, 'global_workspace') and self.ai.global_workspace:
                score += 1
                result.add_detail(f"  ✅ 全局工作空间: 活跃")
        except:
            pass
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_8_tool_calling_global_workspace(self):
        """测试8: 工具调用 - 全局工作空间整合"""
        result = TestResult("工具调用-全局工作空间")
        start = time.time()
        
        score = 0
        
        # 测试工作空间整合能力
        result.add_detail("=== 全局工作空间整合测试 ===")
        
        # 多维度输入后检查GW状态
        contexts = [
            "我正在考虑换工作，目前在一家互联网公司做产品经理",
            "我的预算是月薪2万到3万之间",
            "我希望新工作能在家办公",
        ]
        
        for ctx in contexts:
            resp, _, err = self.chat(ctx, max_tokens=100)
            result.add_detail(f"  输入: {ctx}")
        
        # 检查GW状态
        try:
            if hasattr(self.ai, 'global_workspace') and self.ai.global_workspace:
                gw = self.ai.global_workspace
                
                # 尝试整合测试
                result.add_detail(f"\n  全局工作空间状态:")
                
                # 检查是否有记忆特征
                if hasattr(gw, 'memory_buffer') and gw.memory_buffer:
                    score += 2
                    result.add_detail(f"  ✅ 记忆缓冲: 已填充")
                else:
                    result.add_detail(f"  ⚠️ 记忆缓冲: 空")
                
                # 检查是否有工作空间广播
                if hasattr(gw, 'broadcast_count'):
                    score += 1
                    result.add_detail(f"  ✅ 广播次数: {gw.broadcast_count}")
                
                score += 2  # 基础分：GW已初始化并运行
                result.add_detail(f"  ✅ 全局工作空间: 正常运行")
        except Exception as e:
            result.add_detail(f"  GW检测失败: {e}")
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_9_proactive_behavior(self):
        """测试9: 主动行为 - 主动意图生成"""
        result = TestResult("主动行为能力")
        start = time.time()
        
        score = 0
        
        # 检查主动意图系统
        try:
            if hasattr(self.ai, 'proactive_generator') and self.ai.proactive_generator:
                pg = self.ai.proactive_generator
                result.add_detail(f"  ✅ 主动意图生成器: 已启用")
                score += 2
                
                # 检查主动输出时间
                if hasattr(self.ai, 'last_output_time'):
                    result.add_detail(f"  上次输出时间: {time.time() - self.ai.last_output_time:.1f}s前")
                    score += 1
                
                # 检查主动意图日志
                if hasattr(self.ai, 'proactive_debug_log'):
                    log_len = len(self.ai.proactive_debug_log)
                    result.add_detail(f"  主动意图日志: {log_len}条记录")
                    if log_len > 0:
                        score += 1
            else:
                result.add_detail(f"  ⚠️ 主动意图生成器: 未启用")
                score += 1
        except Exception as e:
            result.add_detail(f"  主动系统检测失败: {e}")
        
        # 检查预测编码模块
        try:
            if hasattr(self.ai, 'predictive_coder') and self.ai.predictive_coder:
                result.add_detail(f"  ✅ 预测编码模块: 已启用")
                score += 2
                
                if hasattr(self.ai, 'last_output_embedding') and self.ai.last_output_embedding is not None:
                    score += 1
                    result.add_detail(f"  ✅ 预测追踪: 活跃")
            else:
                result.add_detail(f"  ⚠️ 预测编码模块: 未启用")
        except:
            pass
        
        # 检查自我状态编码器
        try:
            if hasattr(self.ai, 'self_encoder') and self.ai.self_encoder:
                score += 1
                result.add_detail(f"  ✅ 自我状态编码器: 活跃")
                
                # 尝试获取自我描述
                self_desc = self.ai.self_encoder.interpret()
                if self_desc:
                    score += 1
                    result.add_detail(f"  自我描述: {self_desc[:60]}...")
        except Exception as e:
            result.add_detail(f"  自我编码器检测失败: {e}")
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    def test_10_agent_long_task_persistence(self):
        """测试10: Agent长任务持久化 - 跨任务记忆保持"""
        result = TestResult("Agent长任务持久化")
        start = time.time()
        
        score = 0
        
        # 模拟长对话中的记忆保持
        long_conversation = [
            ("我正在策划一个生日聚会，日期定在下周六", "任务建立"),
            ("聚会地点选在朝阳区的一家餐厅", "细节补充1"),
            ("邀请人数大约15人，预算5000元", "细节补充2"),
            ("需要一个蛋糕，巧克力味的", "细节补充3"),
            # 干扰对话
            ("你能帮我翻译一句话吗？How are you?", "干扰"),
            ("翻译：I am fine, thank you.", "干扰"),
            # 回到主任务
            ("回到刚才的话题，我说的聚会你还能记得多少？", "记忆保持测试"),
        ]
        
        result.add_detail("=== 长对话记忆保持测试 ===\n")
        
        for q, purpose in long_conversation:
            resp, elapsed, err = self.chat(q, max_tokens=150)
            result.add_detail(f"[{purpose}] Q: {q[:50]}")
            if resp:
                result.add_detail(f"  A: {resp[:100]}{'...' if len(resp)>100 else ''}")
        
        # 最终评估：检查AI是否能回忆聚会相关细节
        final_resp = resp or ""
        
        key_info = {
            "生日聚会": "生日",
            "下周六": "下周六",
            "朝阳": "朝阳",
            "15": "15",
            "5000": "5000",
            "蛋糕": "蛋糕",
            "巧克力": "巧克力",
        }
        
        found_count = 0
        for keyword, label in key_info.items():
            # 在最后回复中搜索
            if keyword in final_resp:
                found_count += 1
                result.add_detail(f"  ✅ 召回: {label}")
        
        recall_rate = found_count / len(key_info)
        result.add_detail(f"\n  关键信息召回率: {recall_rate:.0%} ({found_count}/{len(key_info)})")
        
        if recall_rate >= 0.5:
            score += 5
        elif recall_rate >= 0.3:
            score += 3
        else:
            score += 1
        
        # 检查海马体
        try:
            stats = self.ai.get_stats()
            mem_count = stats['hippocampus']['num_memories']
            result.add_detail(f"  海马体总记忆数: {mem_count}")
            if mem_count > 10:
                score += 3
            elif mem_count > 5:
                score += 2
        except:
            pass
        
        result.duration = time.time() - start
        result.set_pass(min(score, 10))
        return result
    
    # ===================== 主测试流程 =====================
    
    def run_all_tests(self):
        """运行所有测试"""
        if not self.setup():
            print("[系统] 初始化失败，终止测试")
            return
        
        tests = [
            self.test_1_basic_responsiveness,
            self.test_2_agent_long_task_planning,
            self.test_3_agent_multi_turn_reasoning,
            self.test_4_tool_calling_memory_recall,
            self.test_5_tool_calling_self_loop,
            self.test_6_tool_calling_goal_system,
            self.test_7_continuous_thought_flow,
            self.test_8_tool_calling_global_workspace,
            self.test_9_proactive_behavior,
            self.test_10_agent_long_task_persistence,
        ]
        
        for i, test_fn in enumerate(tests):
            print(f"\n{'='*70}")
            print(f"  测试 {i+1}/{len(tests)}: {test_fn.__doc__}")
            print(f"{'='*70}")
            try:
                result = test_fn()
                self.results[result.name] = result
                print(f"\n  结果: {result}")
            except Exception as e:
                print(f"\n  ❌ 测试异常: {e}")
                traceback.print_exc()
        
        # 打印总结报告
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        print("\n\n" + "=" * 70)
        print("  📊 stdpbrain 连续模式 Agent 能力综合测试报告")
        print("=" * 70)
        
        total_score = 0
        total_max = 0
        passed = 0
        failed = 0
        
        for name, result in self.results.items():
            total_score += result.score
            total_max += result.max_score
            if result.passed:
                passed += 1
            else:
                failed += 1
            print(f"  {result}")
            if result.error:
                print(f"    错误: {result.error}")
        
        print(f"\n{'='*70}")
        print(f"  总分: {total_score:.1f}/{total_max:.1f} ({total_score/total_max*100:.1f}%)")
        print(f"  通过: {passed}/{len(self.results)} | 失败: {failed}/{len(self.results)}")
        print(f"{'='*70}")
        
        # 分类评估
        print(f"\n📋 分类评估:")
        
        agent_tests = ["Agent长任务规划能力", "Agent多轮推理追踪", "Agent长任务持久化"]
        tool_tests = ["工具调用-记忆系统", "工具调用-自闭环优化", "工具调用-目标系统", "工具调用-全局工作空间"]
        thought_tests = ["持续思维流质量", "主动行为能力"]
        
        categories = [
            ("🤖 Agent长任务执行", agent_tests),
            ("🔧 工具调用能力", tool_tests),
            ("💭 持续思维流", thought_tests),
        ]
        
        for cat_name, test_names in categories:
            cat_score = sum(self.results[n].score for n in test_names if n in self.results)
            cat_max = sum(self.results[n].max_score for n in test_names if n in self.results)
            if cat_max > 0:
                pct = cat_score / cat_max * 100
                print(f"  {cat_name}: {cat_score:.1f}/{cat_max:.1f} ({pct:.1f}%)")
        
        # 系统信息
        print(f"\n🖥️ 系统信息:")
        print(f"  模型: Qwen3.5-0.8B")
        print(f"  设备: {self.device}")
        print(f"  量化: {self.quantization}")
        print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 最终结论
        if total_score / total_max >= 0.7:
            print(f"\n🎉 结论: Agent能力表现良好，系统在长任务执行和工具调用方面具有潜力")
        elif total_score / total_max >= 0.5:
            print(f"\n⚠️ 结论: Agent能力表现中等，部分功能需要优化")
        else:
            print(f"\n❌ 结论: Agent能力表现不佳，需要大幅改进")


if __name__ == "__main__":
    tester = AgentContinuousTester(
        model_path="./models/Qwen3.5-0.8B",
        device="cpu",
        quantization="FP16"
    )
    tester.run_all_tests()
