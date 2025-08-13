// index.js — Sinhala-commented WhatsApp bot with persistent memory + OpenAI→DeepSeek fallback
// - Text AI replies with memory (permanent, saved to file)
// - Voice command ("voice: your text")
// - Voice reply (TTS -> voice note)
// - Image edit options (bw / enhance / cartoon / colorize) + extra prompt support
// - Custom style edit with "style: your description" (photo + style description)
// - Creator info replies
// - Auto fallback to DeepSeek API if OpenAI quota/billing issue

// -------------------- imports --------------------
import pkg from 'whatsapp-web.js';
const { Client, LocalAuth, MessageMedia } = pkg;

import qrcode from 'qrcode-terminal';
import express from 'express';
import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import dotenv from 'dotenv';
dotenv.config();

import googleTTS from 'google-tts-api';
import { v4 as uuidv4 } from 'uuid';
import fetch from 'node-fetch';
import sharp from 'sharp';

import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
ffmpeg.setFfmpegPath(ffmpegPath);

const app = express();
const PORT = process.env.PORT || 3000;

// Main OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// DeepSeek client for fallback
const deepseek = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com"
});

const MEDIA_DIR = path.join(process.cwd(), 'media');
if (!fs.existsSync(MEDIA_DIR)) fs.mkdirSync(MEDIA_DIR);

// -------------------- Persistent history setup --------------------
const HISTORY_FILE = path.join(process.cwd(), 'history.json');
let userSessions = {};

function loadHistory() {
  try {
    if (fs.existsSync(HISTORY_FILE)) {
      const data = fs.readFileSync(HISTORY_FILE, 'utf8').trim();
      if (!data || data === "{}") {
        console.warn("⚠️ history.json empty or default — starting fresh.");
        userSessions = {};
      } else {
        try {
          const parsed = JSON.parse(data);
          if (typeof parsed === 'object' && parsed !== null) {
            userSessions = parsed;
          } else {
            console.warn("⚠️ history.json not an object — resetting sessions.");
            userSessions = {};
          }
        } catch (err) {
          console.error("⚠️ history.json parse error — resetting sessions.", err);
          userSessions = {};
        }
      }
    } else {
      console.log("ℹ️ history.json not found — starting fresh.");
      userSessions = {};
    }
  } catch (err) {
    console.error("❌ Error loading history:", err);
    userSessions = {};
  }
}

function saveHistory() {
  try {
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(userSessions, null, 2), 'utf8');
  } catch (err) {
    console.error("❌ Error saving history:", err);
  }
}

function addMessage(userId, role, text) {
  if (!userSessions[userId]) {
    userSessions[userId] = [];
  }
  userSessions[userId].push({ role, content: text });
  if (userSessions[userId].length > 10) {
    userSessions[userId].shift();
  }
  saveHistory();
}

// -------------------- WhatsApp client --------------------
const client = new Client({
  authStrategy: new LocalAuth(),
  puppeteer: { headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] }
});

const greetedUsers = new Set();

client.on('qr', qr => {
  qrcode.generate(qr, { small: true });
  console.log('QR code generated — scan with WhatsApp (Linked Devices).');
});

client.on('ready', () => {
  console.log('✅ WhatsApp bot ready');
});

// -------------------- helpers --------------------
async function convertMp3ToOgg(mp3Path, oggPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(mp3Path)
      .audioCodec('libopus')
      .audioBitrate('32k')
      .save(oggPath)
      .on('end', () => resolve())
      .on('error', reject);
  });
}

async function generateTTSandConvertToOgg(text) {
  const url = googleTTS.getAudioUrl(text, {
    lang: 'si',
    slow: false,
    host: 'https://translate.google.com',
  });

  const tempMp3 = path.join(MEDIA_DIR, `${uuidv4()}.mp3`);
  const tempOgg = path.join(MEDIA_DIR, `${uuidv4()}.ogg`);

  const resp = await fetch(url);
  const arrayBuffer = await resp.arrayBuffer();
  fs.writeFileSync(tempMp3, Buffer.from(arrayBuffer));

  await convertMp3ToOgg(tempMp3, tempOgg);
  fs.unlinkSync(tempMp3);

  return tempOgg;
}

// -------------------- AI Reply with OpenAI → DeepSeek fallback --------------------
async function getAIReply(userId, userText) {
  addMessage(userId, 'user', userText);
  const system = "You are a helpful assistant. Reply in the same language the user used (Sinhala or English). Be concise and polite.";

  // Try OpenAI first
  try {
    if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY not set");

    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: 'system', content: system },
        ...(userSessions[userId] || [])
      ],
      max_tokens: 400,
      temperature: 0.4
    });

    const aiReply = resp?.choices?.[0]?.message?.content?.trim() ?? "";
    addMessage(userId, 'assistant', aiReply);
    return aiReply;

  } catch (err) {
    // Check if it's quota/billing error
    const msg = String(err?.message || "").toLowerCase();
    const code = err?.error?.code || "";
    const status = err?.status || err?.response?.status;

    const quotaError =
      code === "insufficient_quota" ||
      msg.includes("insufficient_quota") ||
      msg.includes("exceeded your current quota") ||
      status === 402 ||
      (status === 429 && msg.includes("quota"));

    if (!quotaError) {
      throw err; // Not a quota issue
    }

    console.warn("⚠️ OpenAI quota/billing error → switching to DeepSeek...");
  }

  // Fallback to DeepSeek
  try {
    if (!process.env.DEEPSEEK_API_KEY) throw new Error("DEEPSEEK_API_KEY not set");

    const resp = await deepseek.chat.completions.create({
      model: "deepseek-chat",
      messages: [
        { role: 'system', content: system },
        ...(userSessions[userId] || [])
      ],
      max_tokens: 400,
      temperature: 0.4
    });

    const aiReply = resp?.choices?.[0]?.message?.content?.trim() ?? "";
    addMessage(userId, 'assistant', aiReply);
    return aiReply;

  } catch (err) {
    console.error("❌ DeepSeek error:", err);
    throw new Error("Both OpenAI and DeepSeek requests failed.");
  }
}

// -------------------- image edit function with extra prompt --------------------
async function editImageByOption(imagePath, option, extraPrompt = "") {
  let basePrompt = "";
  switch(option) {
    case 'bw':
      basePrompt = "Convert this photo to high-quality black and white, enhance clarity and contrast.";
      break;
    case 'enhance':
      basePrompt = "Enhance the photo: increase resolution, sharpen details, remove noise, keep natural look.";
      break;
    case 'cartoon':
      basePrompt = "Turn this photo into a high-quality cartoon / illustration, vibrant colors, smooth lines.";
      break;
    case 'colorize':
      basePrompt = "Colorize this black and white photo realistically and enhance details.";
      break;
    default:
      basePrompt = "Enhance the photo quality and resolution.";
  }

  if (extraPrompt && extraPrompt.trim().length > 0) {
    basePrompt += ` Also, ${extraPrompt.trim()}`;
  }

  const resp = await openai.images.edit({
    model: "gpt-image-1",
    image: fs.createReadStream(imagePath),
    prompt: basePrompt,
    size: "1024x1024"
  });

  const b64 = resp?.data?.[0]?.b64_json;
  if (!b64) throw new Error('Image edit failed');
  const outPath = path.join(MEDIA_DIR, `${uuidv4()}.png`);
  fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
  return outPath;
}

// -------------------- Message handler --------------------
client.on('message', async message => {
  try {
    const lowerMsg = (message.body || "").toLowerCase().trim();

    // Greet first-time users
    if (!greetedUsers.has(message.from)) {
      await message.reply(
        "🙏🏻 ආයුබෝවන්! මම ජනිතගේ Whatsapp සහයක කෘතිම බුද්ධිය.\n" +
        "☺ ඔබට මට මොන විදිහට උදව් කරන්න පුළුවන්ද?\n\n" +
        "📞 Creator: Janitha Prasad\n" +
        "🌐 සම්බන්ධ වීමට: http://wa.me/94763238609\n" +
        "📄 පණිවිඩය සඳහා මගේ WhatsApp අංකය භාවිතා කරන්න.\n" +
        "☺ ස්තුතියි."
      );
      greetedUsers.add(message.from);
      return;
    }

    // Creator info
    if (
      lowerMsg.includes("oya kawda haduwe") ||
      lowerMsg.includes("oyawa kawda haduwe") ||
      lowerMsg.includes("who created you") ||
      lowerMsg.includes("who made you") ||
      lowerMsg.includes("nirmathru") ||
      lowerMsg.includes("creator")
    ) {
      await message.reply("🛠️ මගේ නිර්මාතෘ: Janitha Prasad ❤️");
      return;
    }

    // Voice command
    if (lowerMsg.startsWith('voice:') || lowerMsg.startsWith('ඔයාවෝස්:')) {
      const text = message.body.split(':').slice(1).join(':').trim();
      if (!text) {
        await message.reply('Voice command format: voice: your text');
        return;
      }

      const aiText = await getAIReply(message.from, text);
      const oggPath = await generateTTSandConvertToOgg(aiText);

      const media = MessageMedia.fromFilePath(oggPath);
      await client.sendMessage(message.from, media, { sendAudioAsVoice: true });
      fs.unlinkSync(oggPath);
      return;
    }

    // -------------------- Image handling --------------------
    if (message.hasMedia) {
      const media = await message.downloadMedia();

      if (media.mimetype && media.mimetype.startsWith('image')) {
        const imgPath = path.join(MEDIA_DIR, `${uuidv4()}.png`);
        const imgBuffer = Buffer.from(media.data, 'base64');
        await sharp(imgBuffer).png().toFile(imgPath);

        const body = (message.body || '').toLowerCase();

        // Custom style mode
        if (body.startsWith('style:')) {
          const customPrompt = message.body.replace(/^style:/i, '').trim();
          if (!customPrompt) {
            await message.reply('⚠️ කරුණාකර "style: your description" format එකෙන් description එක දෙන්න.');
            return;
          }

          await message.reply(`🎨 Applying custom style: "${customPrompt}"... ඉවහල් වෙන්න...`);

          try {
            const resp = await openai.images.edit({
              model: "gpt-image-1",
              image: fs.createReadStream(imgPath),
              prompt: `Apply this style to the photo: ${customPrompt}`,
              size: "1024x1024"
            });

            const b64 = resp?.data?.[0]?.b64_json;
            if (!b64) throw new Error('Image generation failed');
            const outPath = path.join(MEDIA_DIR, `${uuidv4()}.png`);
            fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));

            const finalMedia = MessageMedia.fromFilePath(outPath);
            await message.reply(finalMedia);

            fs.unlinkSync(imgPath);
            fs.unlinkSync(outPath);
          } catch (err) {
            console.error('Custom style edit error:', err);
            await message.reply('❌ Style image edit අසමත් වුණා.');
          }
          return;
        }

        // Predefined modes
        let option = 'enhance';
        if (body.includes('bw')) option = 'bw';
        else if (body.includes('cartoon')) option = 'cartoon';
        else if (body.includes('colorize')) option = 'colorize';
        else if (body.includes('enhance')) option = 'enhance';

        const extraPrompt = body.replace(/bw|cartoon|colorize|enhance/gi, '').trim();

        await message.reply(`🖼️ Image received. Processing (${option}) — ඉවහල් වෙමින් සිටී...`);

        try {
          const editedPath = await editImageByOption(imgPath, option, extraPrompt);
          const editedMedia = MessageMedia.fromFilePath(editedPath);
          await message.reply(editedMedia);
          fs.unlinkSync(imgPath);
          fs.unlinkSync(editedPath);
        } catch (err) {
          console.error('Image edit error:', err);
          await message.reply('සමාවෙන්න, චායාරූප සැකසීම අසමත් වුණා.');
        }
        return;
      }
    }

    // Default: AI text reply with memory
    const aiText = await getAIReply(message.from, message.body || '');
    await message.reply(aiText);

    // Optional voice reply
    if (lowerMsg.includes('reply voice') || lowerMsg.includes('voice reply') || lowerMsg.includes('awazayen')) {
      try {
        const oggPath = await generateTTSandConvertToOgg(aiText);
        const media = MessageMedia.fromFilePath(oggPath);
        await client.sendMessage(message.from, media, { sendAudioAsVoice: true });
        fs.unlinkSync(oggPath);
      } catch (err) {
        console.error('Voice reply error:', err);
      }
    }

  } catch (err) {
    console.error('Message handler error:', err);
    await message.reply('දෝෂයක් සිදු වුනා — නැවත උත්සාහ කරන්න.');
  }
});

// -------------------- start --------------------
loadHistory();
client.initialize();
app.get('/', (req, res) => res.send('WhatsApp bot running'));
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
