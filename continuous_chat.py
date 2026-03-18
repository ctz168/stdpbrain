#!/usr/bin/env python3
"""
持续独白对话模式

特性：
- 持续显示AI的内心独白流
- 用户可以随时观察思维过程
- 用户输入会打断独白并响应
- 类似观察真实人脑的思维流
"""

import time
import threading
import queue
import random
import sys
from datetime import datetime


class ContinuousMonologueChat:
    """持续独白对话模式"""
    
    def __init__(self, ai):
        self.ai = ai
        self.running = True
        self.monologue_queue = queue.Queue()
        self.user_input_queue = queue.Queue()
        self.last_monologue_time = time.time()
        self.monologue_interval = random.uniform(3, 6)  # 3-6秒生成一次独白
        
        # ANSI颜色码
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.GRAY = '\033[90m'
        self.CYAN = '\033[96m'
        self.RESET = '\033[0m'
        
    def monologue_thread(self):
        """后台独白生成线程"""
        print(f"\n{self.BLUE}[独白流] 后台思维流已启动{self.RESET}")
        
        while self.running:
            try:
                # 检查是否有用户输入
                try:
                    user_input = self.user_input_queue.get_nowait()
                    if user_input:
                        # 用户输入打断独白，响应用户
                        self._handle_user_input(user_input)
                        self.last_monologue_time = time.time()
                        continue
                except queue.Empty:
                    pass
                
                # 检查是否该生成独白了
                current_time = time.time()
                if current_time - self.last_monologue_time >= self.monologue_interval:
                    # 生成独白
                    monologue = self._generate_monologue()
                    if monologue:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        # 放入队列显示
                        self.monologue_queue.put((timestamp, monologue))
                    
                    # 更新时间和间隔
                    self.last_monologue_time = current_time
                    self.monologue_interval = random.uniform(3, 6)
                
                # 短暂休眠
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n{self.RED}[错误] 独白线程异常: {e}{self.RESET}")
                time.sleep(1)
        
        print(f"\n{self.BLUE}[独白流] 后台思维流已停止{self.RESET}")
    
    def _generate_monologue(self) -> str:
        """生成独白"""
        try:
            # 使用AI的独白生成
            if hasattr(self.ai, 'think'):
                result = self.ai.think()
                return result.get('monologue', '思考中...')
            elif hasattr(self.ai, '_generate_spontaneous_monologue'):
                return self.ai._generate_spontaneous_monologue(max_tokens=25, temperature=0.85)
            else:
                # 简化独白
                return random.choice([
                    "思维在流动...",
                    "想起了一些事...",
                    "静静地思考...",
                    "脑海中浮现...",
                    "正在回忆..."
                ])
        except Exception as e:
            print(f"\n{self.RED}[错误] 独白生成失败: {e}{self.RESET}")
            return "..."
    
    def _handle_user_input(self, user_input: str):
        """处理用户输入"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 显示用户输入
        print(f"\n{self.GREEN}[{timestamp}] 你：{user_input}{self.RESET}")
        
        # 设置思维种子
        if hasattr(self.ai, 'thought_seed'):
            self.ai.thought_seed = user_input
        
        # 生成回复前的独白（消化输入）
        print(f"{self.GRAY}[思维] 正在消化你的输入...{self.RESET}")
        
        try:
            # 使用chat方法（会生成独白）
            start_time = time.time()
            response = self.ai.chat(user_input, [])
            elapsed = (time.time() - start_time) * 1000
            
            # 显示回复
            print(f"\n{self.YELLOW}[AI] {response}{self.RESET}")
            print(f"{self.BLUE}[耗时：{elapsed:.0f}ms]{self.RESET}")
            
        except Exception as e:
            print(f"\n{self.RED}[错误] 回复生成失败: {e}{self.RESET}")
    
    def input_thread(self):
        """输入线程"""
        while self.running:
            try:
                # 非阻塞输入
                user_input = input()
                if user_input.strip():
                    self.user_input_queue.put(user_input.strip())
                    
                    # 检查退出命令
                    if user_input.strip().lower() in ['quit', 'exit', '退出']:
                        self.running = False
                        break
                        
            except EOFError:
                break
            except KeyboardInterrupt:
                self.running = False
                break
    
    def run(self):
        """运行持续独白对话"""
        print("=" * 70)
        print(f"{self.CYAN}类人脑AI - 持续独白观察模式{self.RESET}")
        print("=" * 70)
        print(f"\n{self.BLUE}特性:{self.RESET}")
        print(f"  * 持续显示AI内心独白流（每3-6秒）")
        print(f"  * 随时观察思维过程")
        print(f"  * 输入消息打断独白并与AI对话")
        print(f"  * 输入 'quit' 或 'exit' 退出")
        print()
        print(f"{self.GRAY}提示: 独白会在后台持续生成，你可以随时输入消息{self.RESET}")
        print(f"{self.GRAY}      或者静静观察AI的思维流{self.RESET}")
        print("=" * 70)
        print()
        
        # 启动线程
        monologue_thread = threading.Thread(target=self.monologue_thread, daemon=True)
        input_thread = threading.Thread(target=self.input_thread, daemon=True)
        
        monologue_thread.start()
        input_thread.start()
        
        # 主显示循环
        try:
            while self.running:
                # 显示独白
                try:
                    timestamp, monologue = self.monologue_queue.get(timeout=0.5)
                    print(f"{self.GRAY}💭 [{timestamp}] {monologue}{self.RESET}")
                except queue.Empty:
                    pass
                
                # 检查线程状态
                if not monologue_thread.is_alive() or not input_thread.is_alive():
                    break
                    
        except KeyboardInterrupt:
            self.running = False
            print(f"\n\n{self.BLUE}[系统] 正在停止...{self.RESET}")
        
        # 等待线程结束
        self.running = False
        time.sleep(0.5)
        
        print(f"\n{self.BLUE}[系统] 正在保存状态...{self.RESET}")
        print(f"{self.CYAN}再见！{self.RESET}\n")


def run_continuous_chat(ai):
    """运行持续独白对话模式"""
    chat = ContinuousMonologueChat(ai)
    chat.run()


if __name__ == "__main__":
    # 测试代码
    print("这是一个模块，请通过 main.py 调用")
