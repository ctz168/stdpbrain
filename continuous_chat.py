"""
持续思维流观察模式 (Continuous Thought Flow Observer)

核心特性:
1. 高刷新小数据: 流式思维显示
2. 流式输出: 打字机效果，可见的思维过程
3. 可打断: 随时输入打断思维流
4. 自闭环优化: 显示自博弈/自评判模式
"""

import time
import random
import sys
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime


class ContinuousThoughtFlowSession:
    """
    持续思维流会话 - 简化版
    """
    
    def __init__(self, ai_interface, config=None):
        self.ai = ai_interface
        self.config = config
        
        # 运行状态
        self.is_running = False
        self.session_start_time = None
        self.cycle_count = 0
        
        # 思维流历史
        self.thought_history: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, str]] = []
        
        # 线程控制
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._current_monologue = ""
        self._monologue_lock = threading.Lock()
        
        # 思维流参数
        self.char_interval = (0.02, 0.05)
        
        # 统计
        self.total_chars = 0
    
    def start(self, duration_minutes: int = 30):
        """启动持续思维流会话"""
        self.is_running = True
        self.session_start_time = time.time()
        self._stop_event.clear()
        self._pause_event.clear()
        
        print("\n" + "=" * 60)
        print("       类人脑AI - 持续思维流观察模式")
        print("=" * 60)
        print("  流式思维显示 - 打字机效果")
        print("  随时输入打断思维流")
        print("  输入 quit 或 exit 退出")
        print("=" * 60 + "\n")
        
        # 启动后台独白线程
        monologue_thread = threading.Thread(target=self._monologue_loop)
        monologue_thread.daemon = True
        monologue_thread.start()
        
        # 主线程处理用户输入 (阻塞式)
        try:
            while self.is_running:
                try:
                    # 阻塞等待用户输入
                    user_input = input()
                    
                    if user_input.strip().lower() in ["quit", "exit"]:
                        print("\n[退出] 正在停止会话...")
                        self.stop()
                        break
                    
                    if user_input.strip():
                        self._handle_user_input(user_input)
                
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\n[中断] 正在停止会话...")
                    self.stop()
                    break
        
        finally:
            self._stop_event.set()
            monologue_thread.join(timeout=1)
            self._print_session_summary()
    
    def _monologue_loop(self):
        """后台内心思维独白生成循环"""
        while self.is_running and not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue
            
            try:
                # 使用统一的内心思维独白引擎
                if hasattr(self.ai, 'inner_thought_engine') and self.ai.inner_thought_engine:
                    # 流式生成内心思维
                    self._display_inner_thought()
                elif hasattr(self.ai, 'model') and hasattr(self.ai.model, 'generate_stream_sync'):
                    # 降级：使用模型流式生成
                    prompt = "思考中... "
                    self._display_stream(prompt)
                else:
                    # 最终降级
                    self._display_simple_thought()
            
            except Exception as e:
                # 静默处理，继续循环
                print(f"\n[思维生成异常: {e}]")
                time.sleep(1)
            
            # 等待下一个周期
            time.sleep(3)
    
    def _display_inner_thought(self):
        """显示内心思维独白"""
        try:
            print("\n[内心思维] ", end="", flush=True)
            
            # 使用流式生成
            generated_any = False
            for char in self.ai.inner_thought_engine.generate_inner_thought(max_tokens=30):
                if self._pause_event.is_set():
                    break
                # 简单的噪音过滤：不显示孤立的特殊符号
                if not generated_any and char in [" ", "|", "<", ">", "-", " "]:
                    continue
                
                print(char, end="", flush=True)
                generated_any = True
                self.total_chars += 1
                time.sleep(random.uniform(*self.char_interval))
            
            if not generated_any:
                print("...", end="", flush=True)
            print()
            
        except Exception as e:
            print(f"\n[内心思维生成错误: {e}]")
    
    def _display_stream(self, prompt: str):
        """流式显示"""
        try:
            print("\n[思维流] ", end="", flush=True)
            for char in self.ai.model.generate_stream_sync(prompt, max_tokens=20, temperature=0.8):
                if self._pause_event.is_set():
                    break
                print(char, end="", flush=True)
                self.total_chars += 1
                time.sleep(random.uniform(*self.char_interval))
            print()
        except Exception as e:
            print(f"\n[流式显示异常: {e}]")
    
    def _display_monologue(self, text: str):
        """显示独白"""
        if not text:
            return
        
        print("\n[思维流] ", end="", flush=True)
        for char in text[:100]:  # 限制长度
            if self._pause_event.is_set():
                break
            print(char, end="", flush=True)
            self.total_chars += 1
            time.sleep(random.uniform(*self.char_interval))
        print()
        
        with self._monologue_lock:
            self._current_monologue = text
            self.thought_history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'content': text[:100]
            })
    
    def _display_simple_thought(self):
        """显示简单思维"""
        thoughts = [
            "思考着今天的事情...",
            "回忆起一些往事...",
            "分析当前的状态...",
            "想象未来的可能性...",
            "整理思绪中...",
        ]
        self._display_monologue(random.choice(thoughts))
    
    def _handle_user_input(self, user_input: str):
        """处理用户输入"""
        # 暂停独白
        self._pause_event.set()
        self.cycle_count += 1
        
        try:
            print("\n" + "-" * 40)
            print("[用户输入]")
            print("-" * 40)
            print(f"用户: {user_input}")
            
            # 判断自闭环模式
            mode = "self_combine"
            mode_names = {
                "self_combine": "自组合",
                "self_game": "自博弈",
                "self_eval": "自评判"
            }
            
            if hasattr(self.ai, 'self_loop') and self.ai.self_loop:
                try:
                    mode = self.ai.self_loop.decide_mode(user_input)
                    print(f"\n[自闭环模式: {mode_names.get(mode, mode)}]")
                except Exception as e:
                    print(f"\n[自闭环模式检测失败: {e}]")
            
            # 显示思考中
            print(f"\nAI: ", end="", flush=True)
            thinking_chars = "思考中..."
            for char in thinking_chars:
                print(char, end="", flush=True)
                time.sleep(0.05)
            print("\r" + " " * 20 + "\r", end="", flush=True)  # 清除
            print(f"AI: ", end="", flush=True)
            
            # 生成回答
            try:
                response = self.ai.chat(
                    user_input,
                    history=self.chat_history[-4:] if self.chat_history else [],
                    max_tokens=200
                )
                
                # 流式输出回答
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(random.uniform(0.01, 0.03))
                
                # 显示自闭环优化统计
                if hasattr(self.ai, 'self_loop') and self.ai.self_loop:
                    try:
                        sl_stats = self.ai.self_loop.get_stats()
                        print(f"\n\n[自闭环统计] 周期={sl_stats['cycle_count']}, 平均准确率={sl_stats['avg_accuracy']:.2f}")
                    except Exception as e:
                        print(f"\n\n[自闭环统计获取失败: {e}]")
                
                print("\n" + "-" * 40 + "\n")
                
                # 记录对话
                self.chat_history.append({"role": "user", "content": user_input})
                self.chat_history.append({"role": "assistant", "content": response})
                
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
            
            except Exception as e:
                print(f"\n[错误] 对话失败: {e}")
                import traceback
                traceback.print_exc()
        
        finally:
            # 恢复独白
            self._pause_event.clear()
    
    def stop(self):
        """停止会话"""
        self.is_running = False
        self._stop_event.set()
    
    def _print_session_summary(self):
        """打印会话总结"""
        duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        print("\n" + "=" * 60)
        print("[ThoughtFlow] 会话结束")
        print("=" * 60)
        print(f"  对话轮数: {len(self.chat_history) // 2}")
        print(f"  思维片段: {len(self.thought_history)}")
        print(f"  总字符数: {self.total_chars}")
        print(f"  持续时间: {duration/60:.1f} 分钟")
        
        # 显示自闭环统计
        if hasattr(self.ai, 'self_loop') and self.ai.self_loop:
            sl_stats = self.ai.self_loop.get_stats()
            print(f"  自闭环周期: {sl_stats['cycle_count']}")
            print(f"  平均准确率: {sl_stats['avg_accuracy']:.2f}")
        
        print("=" * 60)


def run_continuous_chat(ai, duration_minutes: int = 30, **kwargs):
    """运行持续思维流观察模式"""
    session = ContinuousThoughtFlowSession(ai)
    session.start(duration_minutes=duration_minutes)
    return session


if __name__ == "__main__":
    print("持续思维流观察模块")
    print("请通过 main.py --mode continuous 运行")
