# Telnyx Phone Integration Testing Guide

## 📞 Numbers to Call

### Primary Test Number (Telnyx)
**Call this number:**
```
+1 (415) 376-5236
```

This is your Telnyx number that's assigned to the "Alive5 Bridge App".

### Customer-Facing Number (855)
**Your main number:**
```
+1 (855) 551-8858
```

**Current Status:**
- Currently forwards to call center (Posh)
- **To use with AI agent:**
  - **Option 1**: Configure forwarding in your current provider: `855-551-8858` → `+14153765236`
  - **Option 2**: Port the number to Telnyx and assign to "Alive5 Bridge App" (recommended)

---

## ✅ Pre-Testing Checklist

Before testing, ensure:

1. **✅ Webhook URL Updated in Telnyx**
   - Go to: Telnyx Dashboard → Voice API Applications → "Alive5 Bridge App" → Details
   - Webhook URL: `https://18.210.238.67.nip.io/api/telnyx/webhook`
   - Webhook API Version: `API v2`

2. **✅ Backend is Running**
   - Check: `https://18.210.238.67.nip.io/health`
   - Should return: `{"status": "healthy"}`

3. **✅ Worker is Running**
   - Check backend logs: `./logs-worker.ps1`
   - Should show worker is active and ready

4. **✅ LiveKit SIP Trunk is Configured**
   - Trunk ID: `ST_h8MfGhxe3A7R` (from `.env`)
   - SIP Domain: `s95527eskik.sip.livekit.cloud`

---

## 🧪 Testing Steps

### Step 1: Make a Test Call

**Test with Telnyx number (direct):**
1. **Call from any phone:**
   - Dial: `+1 (415) 376-5236`
   - Or: `(415) 376-5236` (if in US)

2. **What should happen:**
   - Call should connect
   - You should hear the AI agent greeting
   - AI should start the conversation

**Test with 855 number (if forwarding configured):**
1. **Call from any phone:**
   - Dial: `+1 (855) 551-8858`
   - Or: `(855) 551-8858` (if in US)

2. **What should happen:**
   - Call forwards to Telnyx number
   - Same flow as above (AI agent greets you)

### Step 2: Monitor Logs

**In separate terminals, run:**

```powershell
# Terminal 1: Backend logs
./logs-backend.ps1

# Terminal 2: Worker logs  
./logs-worker.ps1
```

**What to look for:**

**Backend logs should show:**
```
📞 Telnyx webhook received: call.initiated
📞 Incoming call from +1XXX... to +14153765236
✅ Created LiveKit room: telnyx_call_XXXXX
🌉 Bridging call to LiveKit: sip:telnyx_call_XXXXX@s95527eskik.sip.livekit.cloud
```

**Worker logs should show:**
```
🚀 NEW VOICE SESSION STARTING - Room: telnyx_call_XXXXX
✅ Simple agent started successfully
🎯 SESSION READY
```

### Step 3: Test Conversation

**Try these:**
1. **Greeting**: Wait for AI to greet you
2. **FAQ Question**: Ask "What is Alive5?"
3. **Flow Trigger**: Say something that matches your bot flows (e.g., "I want sales help")
4. **Agent Handoff**: If configured, say "connect me to a person"

---

## 🔍 Troubleshooting

### Issue: Call doesn't connect

**Check:**
1. **Webhook is receiving events:**
   - Look at backend logs for `📞 Telnyx webhook received`
   - If missing → Webhook URL might be wrong

2. **LiveKit room is created:**
   - Check backend logs for `✅ Created LiveKit room`
   - If missing → LiveKit credentials might be wrong

3. **Call is bridged:**
   - Check backend logs for `🌉 Bridging call to LiveKit`
   - If missing → Bridge command might be failing

### Issue: Call connects but no AI response

**Check:**
1. **Worker is running:**
   - Check worker logs for session start
   - If missing → Worker might not be running

2. **Room name matches:**
   - Backend creates: `telnyx_call_{call_control_id}`
   - Worker should join same room
   - Check both logs for matching room names

3. **LiveKit SIP trunk:**
   - Verify trunk is configured in LiveKit dashboard
   - Check trunk allows room name pattern: `telnyx_call_*`

### Issue: Call connects but AI doesn't speak

**Check:**
1. **Voice is configured:**
   - Check worker logs for `🎤 Initializing TTS with voice:`
   - Verify voice ID is valid

2. **Bot flows are loaded:**
   - Check worker logs for `🔧 Loading bot flows`
   - Should show flows loaded successfully

---

## 📊 Expected Flow

```
1. You call +1 (415) 376-5236
   ↓
2. Telnyx receives call
   ↓
3. Telnyx sends webhook to backend: call.initiated
   ↓
4. Backend creates LiveKit room: telnyx_call_XXXXX
   ↓
5. Backend returns commands: answer + bridge
   ↓
6. Telnyx bridges call to LiveKit SIP domain
   ↓
7. LiveKit SIP trunk connects call to room
   ↓
8. Worker joins room and starts AI agent
   ↓
9. AI greets you and conversation begins
```

---

## 🎯 Success Criteria

✅ **Call connects successfully**
✅ **AI agent greets you**
✅ **AI responds to your questions**
✅ **FAQ bot works (if you ask about Alive5)**
✅ **Bot flows work (if you trigger an intent)**

---

## 📝 Notes

- **First call might take 10-15 seconds** to connect (room creation + bridge)
- **Subsequent calls should be faster** (cached connections)
- **If call fails**, check backend logs first (webhook handling)
- **If AI doesn't respond**, check worker logs (agent initialization)

---

## 🆘 Need Help?

If testing fails:
1. Share backend logs (`./logs-backend.ps1`)
2. Share worker logs (`./logs-worker.ps1`)
3. Note what happens when you call (ringing? connects? silence?)

