import asyncio
import sys

from telethon import Button

from Bot import bot, ubot, OWNER_ID, LOG, BOT_TOKEN, SESSION_STRING
from Bot.utils.ffmpeg import check_ffmpeg_installed


async def main():
    LOG.info("Starting Erika-Amano Bot...")

    # Verify ffmpeg/ffprobe are actually installed and runnable before we
    # start accepting encode jobs. This doesn't stop the bot from starting
    # (other features -- auth, admin, stats -- still work without it), but
    # it logs a clear, actionable error immediately instead of only
    # surfacing as an opaque failure on the first user's encode.
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_installed()
    if ffmpeg_ok:
        LOG.info(ffmpeg_msg)
    else:
        LOG.critical(f"FFmpeg check failed: {ffmpeg_msg} Encoding will not work until this is fixed.")

    # Connect bot
    await bot.start(bot_token=BOT_TOKEN)
    LOG.info("Bot client connected.")

    # Connect userbot if session string provided
    if ubot:
        try:
            await ubot.start()
            LOG.info("4GB String Session Connected.")
        except Exception as e:
            LOG.warning(f"Error connecting string session: {e}")

    # Register all handlers
    plugin_modules = [
        ("Bot.plugins.callback", "callback"),
        ("Bot.plugins.encoder", "encoder"),
        ("Bot.plugins.features", "features"),
        ("Bot.plugins.extra", "extra"),
        ("Bot.plugins.user_info", "user_info"),
        ("Bot.plugins.admin", "admin"),
        ("Bot.plugins.eval", "eval"),
    ]
    for module_path, name in plugin_modules:
        try:
            __import__(module_path)
            LOG.info(f"Loaded plugin: {name}")
        except Exception as e:
            LOG.error(f"Failed to load plugin {name}: {e}")

    # Auto-authorize owner in DB
    from Bot.plugins.database.mongo_db import owner_check
    await owner_check()

    # Resume any pending queue items from DB (restart resilience)
    try:
        from Bot.plugins.encoder import resume_pending_queue
        await resume_pending_queue()
        LOG.info("Pending queue check complete.")
    except Exception as e:
        LOG.warning(f"Queue resume failed: {e}")

    try:
        await bot.send_message(
            OWNER_ID,
            "**Bot Started Successfully!**",
            buttons=[
                [Button.inline("ʜᴇʟᴘ", b"answer_help"), Button.inline("ᴅᴇᴠᴇʟᴏᴘᴇʀ", b"answer_about_dev")],
            ],
        )
    except Exception as e:
        LOG.warning(f"Could not send startup message: {e}")

    LOG.info("Bot is running. Press Ctrl+C to stop.")
    await bot.run_until_disconnected()


if __name__ == "__main__":
    try:
        bot.loop.run_until_complete(main())
    except KeyboardInterrupt:
        LOG.info("Bot stopped by user.")
    except Exception as e:
        LOG.critical(f"Fatal error: {e}")
        sys.exit(1)
