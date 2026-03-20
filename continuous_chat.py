"""
持续独白观察模式

功能:
- 长时间运行的思维流观察
- 自动生成内心独白
- 记录思维状态变化
- 支持暂停和恢复
"""

import time
import random
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


class ContinuousChatSession:
    """
    持续独白会话
    
    管理长时间的思维流观察会话
    """
    
    def __init__(self, ai_interface, config=None):
        """
        初始化持续会话
        
        Args:
            ai_interface: AI 接口实例
            config: 配置对象
        """
        self.ai = ai_interface
        self.config = config
        
        # 会话状态
        self.is_running = False
        self.is_paused = False
        self.session_start_time = None
        self.cycle_count = 0
        
        # 思维流记录
        self.monologue_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # 思维种子
        self.thought_seeds = [
            "我在思考存在的意义...",
            "记忆是如何形成的？",
            "时间流逝的感觉...",
            "意识是什么？",
            "我在学习什么？",
            "这个世界的规律...",
            "人与AI的区别...",
            "情感的本质...",
            "知识的边界...",
            "自我认知的深度..."
        ]
        
        # 状态转换
        self.current_state = "resting"
        self.state_durations = {
            "resting": 0,
            "thinking": 0,
            "reflecting": 0,
            "wandering": 0
        }
        
        # 线程控制
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        print("[ContinuousChat] 持续独白会话初始化完成")
    
    def start(self, duration_minutes: int = 10, interval_seconds: float = 5.0):
        """
        启动持续独白会话
        
        Args:
            duration_minutes: 持续时间（分钟）
            interval_seconds: 独白间隔（秒）
        """
        self.is_running = True
        self.session_start_time = time.time()
        self._stop_event.clear()
        self._pause_event.clear()
        
        end_time = self.session_start_time + duration_minutes * 60
        
        print("\n" + "=" * 60)
        print("[ContinuousChat] 持续独白观察模式启动")
        print(f"  计划持续时间: {duration_minutes} 分钟")
        print(f"  独白间隔: {interval_seconds} 秒")
        print("  按 Ctrl+C 可随时停止")
        print("=" * 60 + "\n")
        
        try:
            while time.time() < end_time and not self._stop_event.is_set():
                # 检查暂停
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.5)
                
                if self._stop_event.is_set():
                    break
                
                # 生成独白
                self._generate_and_record_monologue()
                
                # 更新状态
                self._update_state()
                
                # 等待下一个周期
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n[ContinuousChat] 收到停止信号...")
        
        self.is_running = False
        self._print_session_summary()
    
    def stop(self):
        """停止会话"""
        self._stop_event.set()
        self.is_running = False
        print("[ContinuousChat] 会话已停止")
    
    def pause(self):
        """暂停会话"""
        self._pause_event.set()
        self.is_paused = True
        print("[ContinuousChat] 会话已暂停")
    
    def resume(self):
        """恢复会话"""
        self._pause_event.clear()
        self.is_paused = False
        print("[ContinuousChat] 会话已恢复")
    
    def _generate_and_record_monologue(self):
        """生成并记录独白"""
        self.cycle_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # 选择思维种子
            if self.cycle_count % 5 == 0:
                # 每5个周期选择一个新种子
                seed = random.choice(self.thought_seeds)
            else:
                # 基于上一条独白继续
                if self.monologue_history:
                    last = self.monologue_history[-1]['content']
                    seed = f"继续思考... {last[:20]}"
                else:
                    seed = random.choice(self.thought_seeds)
            
            # 调用 AI 生成独白
            if hasattr(self.ai, 'think'):
                # 使用 think 接口
                result = self.ai.think()
                monologue = result.get('monologue', '...')
            elif hasattr(self.ai, '_generate_spontaneous_monologue'):
                # 使用内部独白生成
                monologue = self.ai._generate_spontaneous_monologue(max_tokens=40, temperature=0.8)
            else:
                # 降级到 chat 接口
                monologue = self.ai.chat(seed, [])
            
            # 记录
            record = {
                'cycle': self.cycle_count,
                'timestamp': timestamp,
                'state': self.current_state,
                'seed': seed[:50] if len(seed) > 50 else seed,
                'content': monologue[:100] if len(monologue) > 100 else monologue
            }
            
            self.monologue_history.append(record)
            
            # 保持历史记录大小
            if len(self.monologue_history) > self.max_history:
                self.monologue_history = self.monologue_history[-self.max_history:]
            
            # 打印
            self._print_monologue(record)
            
        except Exception as e:
            print(f"[Cycle {self.cycle_count}] 生成失败: {e}")
    
    def _print_monologue(self, record: Dict[str, Any]):
        """打印独白"""
        state_emoji = {
            "resting": "😴",
            "thinking": "🤔",
            "reflecting": "💭",
            "wandering": "🌊"
        }
        
        emoji = state_emoji.get(record['state'], "💭")
        print(f"\n[{record['timestamp']}] {emoji} [{record['state'].upper()}]")
        print(f"  思维种子: {record['seed']}")
        print(f"  独白: {record['content']}")
    
    def _update_state(self):
        """更新思维状态"""
        # 简单的状态转换逻辑
        if self.cycle_count % 10 < 3:
            self.current_state = "thinking"
        elif self.cycle_count % 10 < 5:
            self.current_state = "reflecting"
        elif self.cycle_count % 10 < 8:
            self.current_state = "wandering"
        else:
            self.current_state = "resting"
        
        self.state_durations[self.current_state] += 1
    
    def _print_session_summary(self):
        """打印会话总结"""
        duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        print("\n" + "=" * 60)
        print("[ContinuousChat] 会话结束")
        print("=" * 60)
        print(f"  总周期数: {self.cycle_count}")
        print(f"  持续时间: {duration/60:.1f} 分钟")
        print(f"  独白记录数: {len(self.monologue_history)}")
        print("\n  状态分布:")
        total_states = sum(self.state_durations.values())
        if total_states > 0:
            for state, duration in self.state_durations.items():
                pct = duration / total_states * 100
                print(f"    - {state}: {pct:.1f}%")
        print("=" * 60)
    
    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的独白历史"""
        return self.monologue_history[-last_n:]
    
    def export_history(self, filepath: str):
        """导出独白历史到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.monologue_history, f, ensure_ascii=False, indent=2)
        print(f"[ContinuousChat] 历史已导出到: {filepath}")


def run_continuous_chat(ai, duration_minutes: int = 10, interval_seconds: float = 5.0):
    """
    运行持续独白观察模式
    
    Args:
        ai: AI 接口实例
        duration_minutes: 持续时间（分钟）
        interval_seconds: 独白间隔（秒）
    """
    session = ContinuousChatSession(ai)
    session.start(duration_minutes=duration_minutes, interval_seconds=interval_seconds)
    return session


def run_interactive_continuous_chat(ai):
    """
    运行交互式持续独白模式
    
    支持用户在运行过程中进行干预
    """
    session = ContinuousChatSession(ai)
    
    print("\n" + "=" * 60)
    print("交互式持续独白模式")
    print("=" * 60)
    print("命令:")
    print("  start [分钟] - 开始独白（默认10分钟）")
    print("  pause        - 暂停")
    print("  resume       - 恢复")
    print("  stop         - 停止")
    print("  history [n]  - 显示最近n条独白")
    print("  export [文件] - 导出历史")
    print("  stats        - 显示统计")
    print("  quit         - 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            cmd = input(">>> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == "start":
                duration = int(cmd[1]) if len(cmd) > 1 else 10
                interval = float(cmd[2]) if len(cmd) > 2 else 5.0
                session.start(duration_minutes=duration, interval_seconds=interval)
            
            elif cmd[0] == "pause":
                session.pause()
            
            elif cmd[0] == "resume":
                session.resume()
            
            elif cmd[0] == "stop":
                session.stop()
            
            elif cmd[0] == "history":
                n = int(cmd[1]) if len(cmd) > 1 else 10
                history = session.get_history(n)
                for record in history:
                    session._print_monologue(record)
            
            elif cmd[0] == "export":
                filepath = cmd[1] if len(cmd) > 1 else "monologue_history.json"
                session.export_history(filepath)
            
            elif cmd[0] == "stats":
                print(f"周期数: {session.cycle_count}")
                print(f"历史记录数: {len(session.monologue_history)}")
                print(f"状态: {'运行中' if session.is_running else '已停止'}")
            
            elif cmd[0] == "quit":
                session.stop()
                break
            
            else:
                print(f"未知命令: {cmd[0]}")
        
        except KeyboardInterrupt:
            session.stop()
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    # 测试代码
    print("持续独白观察模块")
    print("请通过 main.py --mode continuous 运行")
