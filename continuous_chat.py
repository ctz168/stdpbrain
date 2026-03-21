"""
持续思维流观察模式 (Continuous Thought Flow Observer)

核心特性:
1. 高刷新小数据: 0.8秒刷新，每次2-4 tokens
2. 流式输出: 打字机效果，可见的思维过程
3. 增量延续: 新思维基于上一思维延续
4. 状态机驱动: 分析→推理→验证→综合
5. 快速响应: 用户输入时0.3秒内给出反馈
6. 可打断: 随时输入打断思维流
"""

import time
import random
import threading
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime


class ContinuousThoughtFlowSession:
    """
    持续思维流会话 - 高刷新流式模式
    """
    
    def __init__(self, ai_interface, config=None):
        self.ai = ai_interface
        self.config = config
        
        # 运行状态
        self.is_running = False
        self.is_paused = False
        self.session_start_time = None
        self.cycle_count = 0
        
        # 思维流历史
        self.thought_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # 对话历史
        self.chat_history: List[Dict[str, str]] = []
        
        # 线程控制
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._input_event = threading.Event()
        self._user_input = None
        
        # 思维流参数
        self.refresh_cycle = 0.8  # 刷新周期
        self.char_interval = (0.03, 0.1)  # 字符间隔
        
        # 统计
        self.total_chars = 0
        self.total_chunks = 0
    
    def start(self, duration_minutes: int = 30):
        """启动持续思维流会话"""
        self.is_running = True
        self.session_start_time = time.time()
        self._stop_event.clear()
        self._pause_event.clear()
        
        print("\n" + "=" * 60)
        print("       类人脑AI - 持续思维流观察模式")
        print("=" * 60)
        print("  高刷新流式思维 (每0.8秒刷新)")
        print("  打字机效果 - 可见思维过程")
        print("  随时输入打断思维流")
        print("  输入 quit 或 exit 退出")
        print("=" * 60 + "\n")
        
        # 启动思维流线程
        flow_thread = threading.Thread(target=self._thought_flow_loop)
        flow_thread.daemon = True
        flow_thread.start()
        
        # 主线程处理用户输入
        try:
            while self.is_running:
                # 非阻塞输入检测
                user_input = self._non_blocking_input()
                
                if user_input is not None:
                    if user_input.strip().lower() in ["quit", "exit"]:
                        print("\n[退出] 正在停止会话...")
                        self.stop()
                        break
                    
                    if user_input.strip():
                        self._handle_user_input(user_input)
        
        except KeyboardInterrupt:
            print("\n[中断] 正在停止会话...")
            self.stop()
        
        # 等待思维流线程结束
        flow_thread.join(timeout=2)
        
        self._print_session_summary()
    
    def _non_blocking_input(self, timeout: float = 0.1) -> Optional[str]:
        """
        非阻塞输入检测
        
        使用线程实现，超时返回None
        """
        result = [None]
        
        def get_input():
            try:
                result[0] = input()
            except:
                pass
        
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()
        input_thread.join(timeout=timeout)
        
        return result[0]
    
    def _thought_flow_loop(self):
        """思维流生成循环"""
        while self.is_running and not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue
            
            # 生成思维片段
            self._generate_and_display_thought()
            
            # 等待下一个周期
            time.sleep(self.refresh_cycle)
    
    def _generate_and_display_thought(self):
        """生成并显示思维片段"""
        self.cycle_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        try:
            # 获取思维流状态
            if hasattr(self.ai, 'thought_flow_engine') and self.ai.thought_flow_engine:
                state = self.ai.thought_flow_engine.state_machine.state.value
                state_prefix = self.ai.thought_flow_engine.state_machine.get_prefix()
            else:
                state = "thinking"
                state_prefix = ""
            
            # 显示时间戳和状态
            print(f"\n[{timestamp}] [{state.upper()}]")
            print(f"  思维流: ", end="", flush=True)
            
            # 流式生成思维
            chunk_text = ""
            for char in self._stream_thought():
                chunk_text += char
                self.total_chars += 1
                # 打字机效果
                print(char, end="", flush=True)
                time.sleep(random.uniform(*self.char_interval))
            
            print()  # 换行
            
            # 清理思维文本
            chunk_text = self._clean_thought(chunk_text)
            
            # 记录历史
            record = {
                'cycle': self.cycle_count,
                'timestamp': timestamp,
                'state': state,
                'content': chunk_text
            }
            self.thought_history.append(record)
            self.total_chunks += 1
            
            if len(self.thought_history) > self.max_history:
                self.thought_history = self.thought_history[-self.max_history:]
            
            # 更新思维流引擎
            if hasattr(self.ai, 'thought_flow_engine') and self.ai.thought_flow_engine:
                self.ai.thought_flow_engine.update_flow(chunk_text)
        
        except Exception as e:
            print(f"\n[Cycle {self.cycle_count}] 生成失败: {e}")
    
    def _stream_thought(self):
        """流式生成思维"""
        try:
            # 优先使用思维流引擎
            if hasattr(self.ai, 'generate_thought_stream'):
                for item in self.ai.generate_thought_stream(max_chunks=1):
                    if item['type'] == 'char':
                        yield item['content']
                    elif item['type'] == 'chunk_end':
                        break
            elif hasattr(self.ai, 'model') and hasattr(self.ai.model, 'generate_stream_sync'):
                # 使用模型流式生成
                prompt = self._build_simple_prompt()
                for char in self.ai.model.generate_stream_sync(prompt, max_tokens=5, temperature=0.8):
                    yield char
            else:
                # 回退到普通生成
                thought = self.ai.think() if hasattr(self.ai, 'think') else "思考中..."
                text = thought.get('monologue', '...') if isinstance(thought, dict) else str(thought)
                for char in text[:20]:
                    yield char
        
        except Exception as e:
            for char in "...继续...":
                yield char
    
    def _build_simple_prompt(self) -> str:
        """构建简单思维提示"""
        if self.thought_history:
            last = self.thought_history[-1]['content'][-30:]
            return f"思维流延续: {last} → "
        else:
            triggers = ["分析", "推理", "思考", "归纳", "验证"]
            return f"开始思维流: {random.choice(triggers)} "
    
    def _clean_thought(self, text: str) -> str:
        """清理思维文本"""
        if not text:
            return "..."
        
        # 移除标签
        for tag in ['<|im_end|>', '<|im_start|>', '</system>', '<system>']:
            text = text.replace(tag, '')
        
        text = text.strip()
        
        if len(text) > 50:
            text = text[:50] + "..."
        
        return text if text else "..."
    
    def _handle_user_input(self, user_input: str):
        """处理用户输入"""
        # 暂停思维流
        self._pause_event.set()
        
        print("\n" + "-" * 40)
        print("[用户输入打断思维流]")
        print("-" * 40)
        print(f"用户: {user_input}")
        
        # 快速响应
        quick_response = self._get_quick_response(user_input)
        print(f"\nAI: ", end="", flush=True)
        for char in quick_response:
            print(char, end="", flush=True)
            time.sleep(random.uniform(*self.char_interval))
        print()
        
        # 完整回答
        print(f"\nAI: ", end="", flush=True)
        try:
            response = self.ai.chat(
                user_input,
                history=self.chat_history[-4:] if self.chat_history else [],
                max_tokens=150
            )
            
            # 流式输出回答
            for char in response:
                print(char, end="", flush=True)
                time.sleep(random.uniform(0.01, 0.03))
            
            print("\n" + "-" * 40 + "\n")
            
            # 记录对话
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})
            
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
        
        except Exception as e:
            print(f"\n[错误] 对话失败: {e}\n")
        
        # 恢复思维流
        self._pause_event.clear()
    
    def _get_quick_response(self, user_input: str) -> str:
        """获取快速响应"""
        if hasattr(self.ai, 'get_quick_response'):
            return self.ai.get_quick_response(user_input)
        
        # 简化版快速响应
        fillers = {
            "thinking": ["嗯...", "让我想想...", "稍等...", "我想想..."],
            "understanding": ["明白了...", "原来如此...", "好的..."],
            "analyzing": ["分析一下...", "让我看看..."]
        }
        
        text_lower = user_input.lower()
        if any(kw in text_lower for kw in ["是", "对", "好"]):
            return random.choice(fillers["understanding"])
        elif any(kw in text_lower for kw in ["什么", "怎么", "为什么"]):
            return random.choice(fillers["analyzing"])
        else:
            return random.choice(fillers["thinking"])
    
    def stop(self):
        """停止会话"""
        self.is_running = False
        self._stop_event.set()
        print("\n[ThoughtFlow] 会话已停止")
    
    def pause(self):
        """暂停会话"""
        self._pause_event.set()
        self.is_paused = True
        print("[ThoughtFlow] 会话已暂停")
    
    def resume(self):
        """恢复会话"""
        self._pause_event.clear()
        self.is_paused = False
        print("[ThoughtFlow] 会话已恢复")
    
    def _print_session_summary(self):
        """打印会话总结"""
        duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        print("\n" + "=" * 60)
        print("[ThoughtFlow] 会话结束")
        print("=" * 60)
        print(f"  总周期数: {self.cycle_count}")
        print(f"  思维片段: {self.total_chunks}")
        print(f"  总字符数: {self.total_chars}")
        print(f"  持续时间: {duration/60:.1f} 分钟")
        print(f"  对话轮数: {len(self.chat_history) // 2}")
        print(f"  平均速度: {self.total_chars/duration:.1f} 字符/秒" if duration > 0 else "")
        print("=" * 60)
    
    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """获取历史"""
        return self.thought_history[-last_n:]


def run_continuous_chat(ai, duration_minutes: int = 30, **kwargs):
    """运行持续思维流观察模式"""
    session = ContinuousThoughtFlowSession(ai)
    session.start(duration_minutes=duration_minutes)
    return session


if __name__ == "__main__":
    print("持续思维流观察模块")
    print("请通过 main.py --mode continuous 运行")
