#!/usr/bin/env python3
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
print('='*80)
print('类人脑 AI - Telegram Bot (真实模型推理)')
print('='*80)
TOKEN='7983263905:AAFsMuGRdZzWv7KfUaAkJocu0l7LsHrScuc'
print(f'Token: {TOKEN[:20]}...')
try:
   from core.unified_reasoner import UnifiedEnhancedReasoner
   print('加载模型...')
   ai=UnifiedEnhancedReasoner(device='cpu')
    s=ai.get_stats()
   print(f'OK! 模块:{s.get("module_count",0)}/8')
except Exception as e:
   print(f'Fail: {e}')
   ai=None
async def start(u,c): await u.message.reply_text('Bot ready!')
async def h(u,c): await u.message.reply_text('Send message!')
async def a(u,c): await u.message.reply_text('AI: Qwen3.5-0.8B+8 modules')
async def st(u,c):
   if ai: await u.message.reply_text(f'Modules:{ai.get_stats().get("module_count",0)}/8')
    else: await u.message.reply_text('Not loaded')
async def handle(u,c):
   if not u.message.text: return
    await c.bot.send_chat_action(u.chat.id,'typing')
   if ai:
        try:
           r=ai.reason(u.message.text,use_all_enhancements=True)
            await u.message.reply_text(r.text[:4000])
        except Exception as e: await u.message.reply_text(f'Err:{e}')
    else: await u.message.reply_text('AI not running')
from telegram.ext import Application,CommandHandler,MessageHandler,filters
app=Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler('start',start))
app.add_handler(CommandHandler('help',h))
app.add_handler(CommandHandler('about',a))
app.add_handler(CommandHandler('stats',st))
app.add_handler(MessageHandler(filters.TEXT&~filters.COMMAND,handle))
print('Bot ready! Test in Telegram.')
async def main():
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    while True: await asyncio.sleep(1)
try: asyncio.run(main())
except KeyboardInterrupt: print('Stop')
except Exception as e: print(f'Err:{e}')
