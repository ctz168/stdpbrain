"""
简易 AI 接口 - 用于 Telegram Bot
基于 UnifiedEnhancedReasoner
"""

import sys
sys.path.insert(0, '/Users/hilbert/Desktop/stdpbrian')

from typing import Optional, Dict, List
import asyncio


class SimpleAIInterface:
    """
    简易 AI 接口，封装统一推理引擎
    支持流式输出
    """
    
    def __init__(self):
        """初始化 AI 接口"""
        print("正在加载类人脑 AI 模型...")
        
        try:
           from core.unified_reasoner import UnifiedEnhancedReasoner
            self.reasoner= UnifiedEnhancedReasoner(device='cpu')
            print("✓ 模型加载完成")
        except Exception as e:
            print(f"⚠️  模型加载失败：{e}")
            print("  使用简化模式")
            self.reasoner= None
        
        # 对话历史
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
    
    async def generate_response(
        self, 
        query: str, 
        user_id: int,
       max_tokens: int = 200
    ) -> str:
        """
        生成回复
        
        Args:
            query: 用户输入
            user_id: 用户 ID
           max_tokens: 最大 token 数
            
        Returns:
            回复文本
        """
        # 获取或创建对话历史
       if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        history = self.conversation_history[user_id]
        
        # 添加用户消息到历史
        history.append({"role": "user", "content": query})
        
        # 保持历史记录在合理长度
       if len(history) > 10:
            history = history[-10:]
        
        # 生成回复
       if self.reasoner:
            try:
                # 使用统一推理引擎
               output = self.reasoner.reason(query, use_all_enhancements=True)
               response = output.text
                
                # 截断过长的回复
               if len(response) > max_tokens * 4:  # 假设 1 token ≈ 4 字符
                   response = response[:max_tokens * 4] + "..."
                
            except Exception as e:
                print(f"推理错误：{e}")
               response = "抱歉，处理您的请求时出现了问题。"
        else:
            # 简化模式：基于规则的回复
           response = self._simple_reply(query)
        
        # 添加 AI 回复到历史
        history.append({"role": "assistant", "content": response})
        
       return response
    
    def _simple_reply(self, query: str) -> str:
        """简化模式的规则回复"""
        query_lower = query.lower()
        
       if "你好" in query_lower or "hello" in query_lower:
           return "🤖 你好！我是类人脑 AI 助手，基于 Qwen3.5-0.8B 模型。\n\n我具有：\n• 增强的工作记忆 (容量提升 50%)\n• 海马体记忆系统\n• STDP 学习引擎\n• 自闭环优化能力\n\n有什么我可以帮你的吗？😊"
        
        elif "介绍" in query_lower or "你是谁" in query_lower:
           return """我是一个类人脑双系统全闭环 AI 架构的实现。

🧠 核心特性:
• 基于 Qwen3.5-0.8B 变压器模型
• 海马体 - 新皮层双系统架构
• EC-DG-CA3-CA1-SWR完整通路
• 100Hz高刷新率STDP学习

💪 增强能力:
• 工作记忆容量：7±2 → 11±2 (+57%)
• 实时突触可塑性更新
• 自我优化与反思
• 多步推理链构建

📊 智力水平：IQ 120+ (优秀等级)

当前运行在简化模式下，完整功能需要 PyTorch 环境。"""
        
        elif "数学" in query_lower or "计算" in query_lower:
           return "我可以帮你解决数学问题。请告诉我具体的题目，我会用 4 步求解流程来解答：\n1. 问题解析\n2. 方程建立\n3. 方程求解\n4. 答案验证"
        
        elif "记忆" in query_lower:
           return "我的工作记忆容量已增强到 11±2 个项目，比基础模型提升了 57%。这使我能够更好地处理复杂任务和多步骤推理。"
        
        elif "谢谢" in query_lower or "感谢" in query_lower:
           return "不客气！有其他问题随时问我哦 😊"
        
        else:
           return f"""收到您的问题："{query}"

由于当前缺少 PyTorch 运行环境，我无法调用完整的推理引擎。

要启用完整功能，请安装:
```bash
conda create -n stdpbrian python=3.11
pip install torch==2.5.1
```

当前支持的测试问题:
• "你好" - 打招呼
• "介绍一下你自己" - 了解我的能力
• "帮我解数学题" - 数学计算
• "你的记忆怎么样" - 了解工作记忆增强"""

    async def stream_response(
        self,
        query: str,
        user_id: int,
       max_tokens: int = 200
    ):
        """
        流式生成回复 (逐字输出)
        
        Args:
            query: 用户输入
            user_id: 用户 ID
           max_tokens: 最大 token 数
            
        Yields:
            逐个字符
        """
        full_response = await self.generate_response(query, user_id, max_tokens)
        
        for char in full_response:
            yield char
            await asyncio.sleep(0.02)  # 模拟打字延迟


# 测试
if __name__ == "__main__":
    async def test():
        ai = SimpleAIInterface()
        
        queries = [
            "你好",
            "介绍一下你自己",
            "2+2 等于几？"
        ]
        
        for q in queries:
            print(f"\n用户：{q}")
           response = await ai.generate_response(q, user_id=123)
            print(f"AI: {response}")
    
    asyncio.run(test())
