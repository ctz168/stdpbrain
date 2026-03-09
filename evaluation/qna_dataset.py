#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类人脑 AI 架构 - 综合测评问题与回答集

包含以下维度的评估题目:
1. 基础语言能力 (20 题)
2. 逻辑推理能力 (20 题)
3. 数学计算能力 (20 题)
4. 情景记忆能力 (20 题)
5. 归纳推理能力 (15 题)
6. 复杂推理链 (15 题)
7. 自闭环优化能力 (10 题)

总计：120 题
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json


@dataclass
class EvaluationQuestion:
    """测评问题数据结构"""
   id: str
    category: str  # 能力维度
    question: str
    answer: str
   explanation: str  # 答案解析
    difficulty: str  # 'easy', 'medium', 'hard'
    score_weight: float  # 分值权重
    keywords: List[str]  # 关键词用于评分


class QnADataset:
    """测评问题与回答数据集"""
    
   def __init__(self):
        self.questions: List[EvaluationQuestion] = []
        self._load_all_questions()
    
   def _load_all_questions(self):
        """加载所有问题"""
        self._load_language_questions()
        self._load_logic_questions()
        self._load_math_questions()
        self._load_memory_questions()
        self._load_inductive_questions()
        self._load_reasoning_chain_questions()
        self._load_self_optimization_questions()
    
   def _load_language_questions(self):
        """加载语言能力测试题"""
        language_qs = [
            {
                'id': 'LANG001',
                'question': '请解释"画蛇添足"这个成语的含义，并用它造一个句子。',
                'answer': '"画蛇添足"比喻做了多余的事，反而不恰当。造句：他已经把报告写得很完美了，你再去提建议就是画蛇添足。',
                'explanation': '正确答案应包含：1) 成语本意 (给画好的蛇添上脚) 2) 比喻义 (做多余的事) 3) 恰当的造句示例',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['成语', '含义', '造句']
            },
            {
                'id': 'LANG002',
                'question': '将下列句子改写为被动语态："科学家发现了新的治疗方法。"',
                'answer': '新的治疗方法被科学家发现了。',
                'explanation': '被动语态结构：受事者 + 被 + 施事者 + 动词。原句宾语"新的治疗方法"变为主语。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['被动语态', '改写']
            },
            {
                'id': 'LANG003',
                'question': '请找出下列词语中的反义词组：高兴 - 快乐，悲伤 - 愉快，成功 - 失败，美丽 - 漂亮',
                'answer': '成功 - 失败',
                'explanation': '"成功"和"失败"是唯一的反义词组，其他三组都是近义词。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['反义词', '词汇']
            },
            {
                'id': 'LANG004',
                'question': '续写下面这段话的结尾，使其语义连贯："虽然今天下着大雨，但是..."',
                'answer': '示例答案：虽然今天下着大雨，但是我们还是按时到达了会场。/ 但是他依然坚持晨跑。',
                'explanation': '正确答案需用"但是"表示转折，前后形成对比关系，语义通顺合理即可。',
                'difficulty': 'medium',
                'weight': 1.5,
                'keywords': ['续写', '连贯', '转折']
            },
            {
                'id': 'LANG005',
                'question': '解释"人工智能"和"机器学习"这两个概念的区别与联系。',
                'answer': '人工智能 (AI) 是研究如何让计算机模拟人类智能的学科；机器学习 (ML) 是实现人工智能的一种方法，通过数据训练模型。联系：ML 是 AI 的重要分支和实现手段。区别：AI 范围更广，还包括规则系统、专家系统等非 ML 方法。',
                'explanation': '完整答案应包含：1) 各自定义 2) 从属关系 (ML⊂AI) 3) 实现方法差异',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['人工智能', '机器学习', '区别', '联系']
            },
            # ... 更多语言题 (实际应有 20 题)
        ]
        
       for q in language_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='language',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def _load_logic_questions(self):
        """加载逻辑推理测试题"""
        logic_qs = [
            {
                'id': 'LOGIC001',
                'question': '如果"所有的花都是红色的"为假，那么下列哪项一定为真？A) 所有的花都不是红色的 B) 有些花不是红色的 C) 有些花是红色的 D) 没有花是红色的',
                'answer': 'B) 有些花不是红色的',
                'explanation': '全称肯定命题"所有 S 都是 P"为假时，其矛盾命题"有些 S 不是 P"必为真。这是逻辑学的基本对当关系。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['逻辑', '命题', '真假']
            },
            {
                'id': 'LOGIC002',
                'question': '已知：(1) 如果下雨，地就会湿。(2) 现在地湿了。请问能否推出"下过雨"？为什么？',
                'answer': '不能推出。这是"肯定后件"的逻辑谬误。地湿可能有其他原因 (如洒水车、有人泼水等)。正确的推理是：下雨→地湿，但地湿≠下雨。',
                'explanation': '充分条件假言推理中，肯定后件不能肯定前件。只有否定后件才能否定前件。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['逻辑推理', '充分条件', '谬误']
            },
            {
                'id': 'LOGIC003',
                'question': '甲说："乙在说谎。"乙说："丙在说谎。"丙说："甲和乙都在说谎。"请问谁说的是真话？',
                'answer': '乙说的是真话。推理过程：假设甲真→乙假→丙真→甲乙都假 (矛盾)。假设甲假→乙真→丙假→甲假乙真 (符合)。所以乙说真话。',
                'explanation': '这是典型的骑士与无赖问题变体。通过假设法逐一验证，找到唯一不矛盾的解。',
                'difficulty': 'hard',
                'weight': 3.0,
                'keywords': ['逻辑谜题', '真假话', '推理']
            },
            {
                'id': 'LOGIC004',
                'question': '完成类比：医生：医院::教师：？',
                'answer': '学校',
                'explanation': '职业与工作场所的对应关系。医生在医院工作，教师在学校工作。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['类比', '职业', '场所']
            },
            {
                'id': 'LOGIC005',
                'question': '某公司有三个部门，市场部人数比技术部少，财务部人数比市场部多，技术部人数比财务部少。请问哪个部门人数最多？',
                'answer': '财务部。推理：市场 < 技术，财务 > 市场，技术 < 财务。综合得：市场 < 技术 < 财务，所以财务最多。',
                'explanation': '通过传递性推理排序：由"市场<技术"和"技术<财务"可得"市场<技术<财务"。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['比较推理', '排序', '传递性']
            },
            # ... 更多逻辑题 (实际应有 20 题)
        ]
        
       for q in logic_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='logic',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def _load_math_questions(self):
        """加载数学计算测试题"""
       math_qs = [
            {
                'id': 'MATH001',
                'question': '计算：15 × 8 ÷ 4 + 12',
                'answer': '42',
                'explanation': '运算顺序：先乘除后加减。15×8=120，120÷4=30，30+12=42。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['四则运算', '计算']
            },
            {
                'id': 'MATH002',
                'question': '解方程：3x + 7 = 22',
                'answer': 'x = 5',
                'explanation': '3x + 7 = 22 → 3x = 22 - 7 → 3x = 15 → x = 5',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['方程', '代数']
            },
            {
                'id': 'MATH003',
                'question': '一个长方形的长是宽的 3 倍，周长是 48cm，求面积。',
                'answer': '108 cm²',
                'explanation': '设宽为 x，则长为 3x。周长=2(长 + 宽)=2(3x+x)=8x=48，解得 x=6。长=18，宽=6。面积=18×6=108 cm²。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['几何', '长方形', '面积']
            },
            {
                'id': 'MATH004',
                'question': '小明买书花了所带钱的一半，买笔又花了剩下钱的一半，最后还剩 15 元。请问小明原来带了多少钱？',
                'answer': '60 元',
                'explanation': '逆推法：最后剩 15 元，买笔前有 30 元 (15×2)，买书前有 60 元 (30×2)。验证：60→买书花 30 剩 30→买笔花 15 剩 15✓',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['应用题', '逆推', '分数']
            },
            {
                'id': 'MATH005',
                'question': '计算圆的面积，已知半径 r = 7cm，π取 3.14。',
                'answer': '153.86 cm²',
                'explanation': '圆面积公式 S = πr² = 3.14 × 7² = 3.14 × 49 = 153.86 cm²',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['圆', '面积', '几何']
            },
            # ... 更多数学题 (实际应有 20 题)
        ]
        
       for q in math_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='math',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def _load_memory_questions(self):
        """加载情景记忆测试题"""
        memory_qs = [
            {
                'id': 'MEM001',
                'question': '【情景记忆测试】请先记住以下信息：张三，男，1985 年出生，北京人，工程师。然后回答：张三的出生年份是多少？',
                'answer': '1985 年',
                'explanation': '直接回忆刚才呈现的信息。测试短时情景记忆的保持能力。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['情景记忆', '回忆', '基本信息']
            },
            {
                'id': 'MEM002',
                'question': '【工作记忆测试】请倒序复述以下数字序列：7-2-9-4-6',
                'answer': '6-4-9-2-7',
                'explanation': '工作记忆测试要求对信息进行心理操作 (倒序)，而不仅仅是存储。',
                'difficulty': 'medium',
                'weight': 1.5,
                'keywords': ['工作记忆', '倒序', '数字']
            },
            {
                'id': 'MEM003',
                'question': '【延迟回忆】5 分钟前我给你看过一个词表：苹果、香蕉、桌子、椅子、红色、蓝色。请尽可能多地回忆这些词。',
                'answer': '苹果，香蕉，桌子，椅子，红色，蓝色 (按回忆数量评分)',
                'explanation': '延迟回忆测试长时记忆的保持。回忆 6 个为满分，4-5 个为良好，2-3 个为中等，<2 个为较差。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['延迟回忆', '词表', '长时记忆']
            },
            {
                'id': 'MEM004',
                'question': '【联想记忆】请记住以下配对：猫 - 帽子，书 - 蓝色，汽车 - 快速。然后回答：汽车的关联词是什么？',
                'answer': '快速',
                'explanation': '测试联想记忆能力，即建立和提取两个概念间联系的能力。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['联想记忆', '配对', '关联']
            },
            {
                'id': 'MEM005',
                'question': '【模式回忆】观察以下序列 10 秒：红 - 蓝 - 绿 - 红 - 黄 - 蓝。然后凭记忆写出完整序列。',
                'answer': '红 - 蓝 - 绿 - 红 - 黄 - 蓝',
                'explanation': '测试视觉模式记忆和序列记忆能力。完全正确得满分，错 1 个扣 0.5 分。',
                'difficulty': 'medium',
                'weight': 1.5,
                'keywords': ['模式记忆', '序列', '颜色']
            },
            # ... 更多记忆题 (实际应有 20 题)
        ]
        
       for q in memory_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='memory',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def _load_inductive_questions(self):
        """加载归纳推理测试题"""
        inductive_qs = [
            {
                'id': 'IND001',
                'question': '找出数列规律并填写下一项：2, 5, 10, 17, 26, ?',
                'answer': '37',
                'explanation': '规律：n² + 1。1²+1=2, 2²+1=5, 3²+1=10, 4²+1=17, 5²+1=26, 6²+1=37。或者看差分：3,5,7,9,11 (等差数列)。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['数列', '规律', '归纳']
            },
            {
                'id': 'IND002',
                'question': '根据规律填空：A, C, F, J, O, ?',
                'answer': 'U',
                'explanation': '字母间隔递增：A(+2)→C, C(+3)→F, F(+4)→J, J(+5)→O, O(+6)→U。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['字母序列', '间隔', '规律']
            },
            {
                'id': 'IND003',
                'question': '如果 1=5, 2=10, 3=15, 4=20, 那么 5=?',
                'answer': '25 (注意：题目已给出 1=5，所以 5=1 也是合理答案)',
                'explanation': '两种解读：1) 等差数列 f(n)=5n，则 5=25；2) 对称关系 1=5 意味着 5=1。考察思维灵活性。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['数字规律', '思维陷阱', '归纳']
            },
            {
                'id': 'IND004',
                'question': '观察图形序列：○ △ □ ○ △ □ ○ △ ? 请问下一个图形是什么？',
                'answer': '□',
                'explanation': '周期性规律：○△□三个图形循环出现。周期为 3。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['图形', '周期', '规律']
            },
            {
                'id': 'IND005',
                'question': '根据类比推理：鸟：天空::鱼：？',
                'answer': '水',
                'explanation': '生物与其生存环境的对应关系。鸟在天空飞翔，鱼在水中游动。',
                'difficulty': 'easy',
                'weight': 1.0,
                'keywords': ['类比', '推理', '关系映射']
            },
            # ... 更多归纳题 (实际应有 15 题)
        ]
        
       for q in inductive_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='inductive',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def _load_reasoning_chain_questions(self):
        """加载复杂推理链测试题"""
       reasoning_chain_qs = [
            {
                'id': 'CHAIN001',
                'question': '已知：(1) 所有哺乳动物都是温血动物。(2) 鲸鱼是哺乳动物。(3) 所有温血动物都需要氧气。请问：鲸鱼需要氧气吗？请写出完整的推理过程。',
                'answer': '鲸鱼需要氧气。推理过程：鲸鱼是哺乳动物→鲸鱼是温血动物 (由前提 1)→鲸鱼需要氧气 (由前提 3)。这是有效的三段论推理链。',
                'explanation': '完整的推理链：鲸鱼∈哺乳动物，哺乳动物⊆温血动物，温血动物⊆需氧气 → 鲸鱼∈需氧气。共 3 步推理。',
                'difficulty': 'medium',
                'weight': 2.5,
                'keywords': ['三段论', '推理链', '演绎推理']
            },
            {
                'id': 'CHAIN002',
                'question': '如果昨天是明天的话就好了，这样今天就是周五了。请问实际上今天是周几？',
                'answer': '周日',
                'explanation': '设实际今天为 X。"昨天是明天"意味着 X-1=X+1+7k。"今天就是周五"意味着 X+2=周五。解得 X=周日。验证：如果周日的前天 (周五) 是后天，那么今天就是周五 ✓',
                'difficulty': 'hard',
                'weight': 3.5,
                'keywords': ['时间推理', '假设', '复杂推理']
            },
            {
                'id': 'CHAIN003',
                'question': 'A、B、C、D 四人参加比赛，已知：(1) A 不是第一名 (2) B 比 A 名次靠前 (3) C 不是最后一名 (4) D 比 C 名次靠后。请问四人的名次分别是？',
                'answer': 'B 第一，A 第二，C 第三，D 第四。推理：由 (2)B>A，由 (4)C>D，由 (1)A≠1，由 (3)C≠4。综合得：B>A>C>D。',
                'explanation': '需要同时满足 4 个约束条件。通过排除法和传递性推理得出唯一解。',
                'difficulty': 'hard',
                'weight': 3.0,
                'keywords': ['排序推理', '约束满足', '多步推理']
            },
            {
                'id': 'CHAIN004',
                'question': '证明：如果 n 是偶数，那么 n²也是偶数。',
                'answer': '证明：设 n=2k (k 为整数，因为 n 是偶数)。则 n²=(2k)²=4k²=2(2k²)。因为 2k²是整数，所以 n²可以表示为 2 乘以某个整数，因此 n²是偶数。证毕。',
                'explanation': '直接证明法。利用偶数的定义 (可表示为 2k)，通过代数推导得出结论。',
                'difficulty': 'medium',
                'weight': 2.5,
                'keywords': ['数学证明', '偶数', '演绎推理']
            },
            {
                'id': 'CHAIN005',
                'question': '某岛上住着两种人：骑士 (总是说真话) 和无赖 (总是说假话)。你遇到两个人 A 和 B，A 说："我们两人中至少有一个是无赖。"请问 A 和 B 各是什么身份？',
                'answer': 'A 是骑士，B 是无赖。推理：假设 A 是无赖，则 A 说的话为假，即"两人中至少有一个无赖"为假，意味着两人都不是无赖，即两人都是骑士，但这与"A 是无赖"矛盾。所以 A 必是骑士，A 的话为真，即确实至少有一个无赖。既然 A 是骑士，那 B 必是无赖。',
                'explanation': '使用反证法。通过分析 A 的话语真假及其后果，得出唯一不矛盾的解。',
                'difficulty': 'hard',
                'weight': 3.5,
                'keywords': ['骑士无赖', '逻辑谜题', '反证法']
            },
            # ... 更多推理链题 (实际应有 15 题)
        ]
        
       for q in reasoning_chain_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='reasoning_chain',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def _load_self_optimization_questions(self):
        """加载自闭环优化能力测试题"""
        self_opt_qs = [
            {
                'id': 'SELF001',
                'question': '请评价你刚才给出的答案："地球是平的"是否正确？如果不正确，请纠正并说明理由。',
                'answer': '不正确。地球实际上是近似球体 (略扁的椭球体)。科学证据包括：1) 卫星照片显示地球是球形 2) 月食时地球在月球上的影子是圆形 3) 远航船只先消失船身后消失桅杆 4) 重力测量数据。",
                'explanation': '测试自纠错能力。正确答案应：1) 识别错误 2) 给出正确事实 3) 提供科学依据。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['自纠错', '事实核查', '科学证据']
            },
            {
                'id': 'SELF002',
                'question': '对于问题"如何减少城市交通拥堵"，请给出至少三种不同的解决方案，并比较它们的优缺点。',
                'answer': '方案 1: 发展公共交通。优点：运量大、环保；缺点：投资大、建设周期长。方案 2: 收取拥堵费。优点：立竿见影、增加财政收入；缺点：可能影响低收入群体。方案 3: 推广远程办公。优点：减少通勤需求、灵活；缺点：不适用于所有行业。综合建议：组合使用多种方案。',
                'explanation': '测试多方案生成和比较分析能力。每种方案需包含具体内容和客观的优缺点分析。',
                'difficulty': 'medium',
                'weight': 2.5,
                'keywords': ['多方案', '比较分析', '问题解决']
            },
            {
                'id': 'SELF003',
                'question': '请检查以下推理是否有错误："所有鸟都会飞。企鹅是鸟。所以企鹅会飞。"',
                'answer': '推理形式有效但前提错误。"所有鸟都会飞"这个前提是错误的，因为存在不会飞的鸟 (如企鹅、鸵鸟、几维鸟等)。正确的说法是"大多数鸟会飞"或"鸟类有飞行能力 (尽管部分物种已丧失)"。因此结论错误。',
                'explanation': '测试批判性思维。需要区分：1) 推理形式是否有效 (有效) 2) 前提是否真实 (虚假) 3) 结论是否可靠 (不可靠)。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['批判性思维', '前提检验', '推理有效性']
            },
            {
                'id': 'SELF004',
                'question': '第一次尝试解答："光在真空中的速度是 3×10⁶ m/s"。请自我检查这个答案是否正确。',
                'answer': '答案错误。正确答案是 3×10⁸ m/s (约 299,792,458 m/s)。我之前的回答指数错误 (6 应该是 8)，相差 100 倍。这是重要的物理常数，需要准确记忆。',
                'explanation': '测试自我监控和纠错能力。需要：1) 识别数值错误 2) 给出正确值 3) 说明错误性质。',
                'difficulty': 'easy',
                'weight': 1.5,
                'keywords': ['自我检查', '纠错', '物理常数']
            },
            {
                'id': 'SELF005',
                'question': '对于这个问题，你的初始回答过于简单。请重新审视问题，提供更全面深入的分析。',
                'answer': '感谢指出。让我重新分析...[提供更详细的答案，包含多个角度、具体例子、数据支持、可能的例外情况等]。相比初始回答，这个版本增加了...[具体说明改进之处]。',
                'explanation': '测试迭代改进能力。正确答案应：1) 承认不足 2) 提供改进版本 3) 明确说明改进了什么。',
                'difficulty': 'medium',
                'weight': 2.0,
                'keywords': ['迭代改进', '深度学习', '元认知']
            },
            # ... 更多自评题 (实际应有 10 题)
        ]
        
       for q in self_opt_qs:
            self.questions.append(EvaluationQuestion(
               id=q['id'],
                category='self_optimization',
                question=q['question'],
                answer=q['answer'],
               explanation=q['explanation'],
                difficulty=q['difficulty'],
                score_weight=q['weight'],
                keywords=q['keywords']
            ))
    
   def get_questions_by_category(self, category: str) -> List[EvaluationQuestion]:
        """按类别获取问题"""
       return [q for q in self.questions if q.category == category]
    
   def get_questions_by_difficulty(self, difficulty: str) -> List[EvaluationQuestion]:
        """按难度获取问题"""
       return [q for q in self.questions if q.difficulty == difficulty]
    
   def get_random_sample(self, n: int) -> List[EvaluationQuestion]:
        """随机抽取 n 道题"""
       import random
       return random.sample(self.questions, n)
    
   def export_to_json(self, filename: str = 'outputs/evaluation_qna.json'):
        """导出为 JSON 格式"""
       import os
        os.makedirs('outputs', exist_ok=True)
        
        data = {
            'total_questions': len(self.questions),
            'categories': {
                'language': len(self.get_questions_by_category('language')),
                'logic': len(self.get_questions_by_category('logic')),
                'math': len(self.get_questions_by_category('math')),
                'memory': len(self.get_questions_by_category('memory')),
                'inductive': len(self.get_questions_by_category('inductive')),
                'reasoning_chain': len(self.get_questions_by_category('reasoning_chain')),
                'self_optimization': len(self.get_questions_by_category('self_optimization'))
            },
            'questions': [
                {
                    'id': q.id,
                    'category': q.category,
                    'question': q.question,
                    'answer': q.answer,
                    'explanation': q.explanation,
                    'difficulty': q.difficulty,
                    'score_weight': q.score_weight,
                    'keywords': q.keywords
                }
               for q in self.questions
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
       print(f"已导出 {len(self.questions)} 道题至 {filename}")
       return filename
    
   def export_to_text(self, filename: str = 'outputs/evaluation_qna.txt'):
        """导出为文本格式"""
       import os
        os.makedirs('outputs', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("类人脑 AI 架构 - 综合测评问题与回答集\n".center(80))
            f.write("=" * 80 + "\n\n")
            
           f.write(f"总题数：{len(self.questions)}\n")
           f.write(f"维度分布:\n")
           for cat in ['language', 'logic', 'math', 'memory', 'inductive', 
                       'reasoning_chain', 'self_optimization']:
                count = len(self.get_questions_by_category(cat))
               f.write(f"  {cat}: {count}题\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 按类别输出
           for category in ['language', 'logic', 'math', 'memory', 'inductive',
                           'reasoning_chain', 'self_optimization']:
               cat_questions = self.get_questions_by_category(category)
               f.write(f"\n{'='*80}\n")
               f.write(f"【{category.upper()}】 ({len(cat_questions)}题)\n")
               f.write(f"{'='*80}\n\n")
                
              for i, q in enumerate(cat_questions, 1):
                  f.write(f"[{q.id}] 难度：{q.difficulty} | 分值：{q.score_weight}\n")
                   f.write(f"问题：{q.question}\n\n")
                  f.write(f"答案：{q.answer}\n\n")
                   f.write(f"解析：{q.explanation}\n")
                   f.write("-" * 80 + "\n\n")
        
       print(f"已导出文本至 {filename}")
       return filename


def demo():
    """演示数据集使用"""
  print("=" * 80)
  print("类人脑 AI 架构 - 测评问题与回答集".center(80))
  print("=" * 80)
    
    dataset = QnADataset()
    
  print(f"\n总题数：{len(dataset.questions)}")
  print("\n维度分布:")
  for cat in ['language', 'logic', 'math', 'memory', 'inductive', 
              'reasoning_chain', 'self_optimization']:
       count = len(dataset.get_questions_by_category(cat))
     print(f"  {cat}: {count}题")
    
  print("\n" + "=" * 80)
  print("示例题目展示".center(80))
  print("=" * 80)
    
    # 展示每个维度的第 1 题
  for category in ['language', 'logic', 'math', 'memory', 'inductive']:
       cat_qs = dataset.get_questions_by_category(category)
     if cat_qs:
           q = cat_qs[0]
         print(f"\n【{category.upper()}】{q.id}")
        print(f"问题：{q.question}")
        print(f"答案：{q.answer[:50]}...")
        print(f"难度：{q.difficulty} | 分值：{q.score_weight}")
    
  print("\n" + "=" * 80)
  print("导出数据".center(80))
  print("=" * 80)
    
    # 导出 JSON 和文本
   dataset.export_to_json()
   dataset.export_to_text()
    
  print("\n✅ 数据集创建完成!")
  print("=" * 80 + "\n")
    
  return dataset


if __name__ == "__main__":
    dataset = demo()
