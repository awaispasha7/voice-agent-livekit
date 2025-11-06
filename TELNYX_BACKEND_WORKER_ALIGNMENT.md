# Backend & Worker Alignment for Telnyx Phone Calls

## ✅ Alignment Status: **SYNCHRONIZED**

Both backend and worker are now properly aligned for Telnyx phone call functionality.

---

## 🔄 Call Flow

```
1. Phone Call → Telnyx
   ↓
2. Telnyx sends webhook: call.initiated
   ↓
3. Backend receives webhook → Creates LiveKit room
   ↓
4. Backend stores session data with phone call metadata
   ↓
5. Backend returns commands: answer + dial (bridge to LiveKit)
   ↓
6. Telnyx bridges call to LiveKit SIP domain
   ↓
7. Worker detects phone call → Joins room
   ↓
8. Worker loads session data → Starts AI agent
   ↓
9. AI agent greets caller → Conversation begins
```

---

## 📋 Backend Responsibilities (`main.py`)

### Webhook Handler (`/api/telnyx/webhook`)
- ✅ Receives `call.initiated` event from Telnyx
- ✅ Creates LiveKit room: `telnyx_call_{call_control_id}`
- ✅ Stores session data with:
  ```python
  {
    "room_name": "telnyx_call_XXXXX",
    "user_name": "Caller_{caller_number}",
    "call_control_id": "...",
    "caller_number": "+1XXX...",
    "called_number": "+14153765236",
    "user_data": {
      "botchain_name": "voice-1",
      "org_name": "alive5stage0",
      "faq_isVoice": True,
      "selected_voice": "...",
      "faq_bot_id": "...",
      "special_instructions": "",
      "source": "telnyx_phone"  # ← Key identifier
    }
  }
  ```
- ✅ Returns Telnyx commands: `answer` + `dial` (bridge to LiveKit)

### Session Storage
- ✅ Room name pattern: `telnyx_call_{call_control_id}`
- ✅ Source identifier: `"source": "telnyx_phone"`
- ✅ Default bot configuration from `.env`:
  - `TELNYX_DEFAULT_BOTCHAIN`
  - `TELNYX_DEFAULT_ORG`
  - `TELNYX_DEFAULT_FAQ_BOT`

---

## 🤖 Worker Responsibilities (`worker.py`)

### Phone Call Detection
- ✅ Detects phone calls by room name: `ctx.room.name.startswith("telnyx_call_")`
- ✅ Confirms via session data: `user_data.get("source") == "telnyx_phone"`

### Session Data Loading
- ✅ Fetches session from backend: `GET /api/sessions/{room_name}`
- ✅ Extracts configuration:
  - `botchain_name` (default: "voice-1")
  - `org_name` (default: "alive5stage0")
  - `faq_isVoice` (default: True)
  - `special_instructions` (default: "")
  - `selected_voice` (from session or default)

### Livechat Handling
- ✅ **Skips livechat initialization** for phone calls
- ✅ **Skips livechat cleanup** on session end for phone calls
- ✅ Only initializes livechat for web sessions

### Agent Initialization
- ✅ Creates `SimpleVoiceAgent` with correct configuration
- ✅ Sets `faq_isVoice` flag
- ✅ Loads bot flows and starts conversation

---

## 🔑 Key Alignment Points

### 1. Room Name Convention
- **Backend creates**: `telnyx_call_{call_control_id}`
- **Worker detects**: `room_name.startswith("telnyx_call_")`
- ✅ **ALIGNED**

### 2. Session Data Structure
- **Backend stores**: `user_data.source = "telnyx_phone"`
- **Worker checks**: `user_data.get("source") == "telnyx_phone"`
- ✅ **ALIGNED**

### 3. Configuration Defaults
- **Backend uses**: `TELNYX_DEFAULT_*` from `.env`
- **Worker uses**: Same defaults if session fetch fails
- ✅ **ALIGNED**

### 4. Livechat Integration
- **Backend**: No livechat for phone calls
- **Worker**: Skips livechat init/cleanup for phone calls
- ✅ **ALIGNED**

### 5. Bot Configuration
- **Backend**: Stores botchain_name, org_name, faq_bot_id in session
- **Worker**: Loads from session or uses defaults
- ✅ **ALIGNED**

---

## 🧪 Testing Checklist

When testing phone calls, verify:

- [ ] Backend receives webhook: `📞 Telnyx webhook received: call.initiated`
- [ ] Backend creates room: `✅ Created LiveKit room: telnyx_call_XXXXX`
- [ ] Backend stores session: Session data includes `"source": "telnyx_phone"`
- [ ] Worker detects phone call: `📞 Phone call detected - skipping livechat initialization`
- [ ] Worker loads session: Session data fetched successfully
- [ ] Worker starts agent: `🚀 NEW VOICE SESSION STARTING`
- [ ] AI greets caller: Agent speaks greeting
- [ ] Conversation works: AI responds to caller's questions

---

## 🐛 Troubleshooting

### Issue: Worker doesn't detect phone call
**Check:**
- Room name starts with `telnyx_call_`
- Session data has `"source": "telnyx_phone"`

### Issue: Livechat initialization fails for phone calls
**Fix:** Already handled - worker skips livechat for phone calls

### Issue: Session data not found
**Check:**
- Backend created session before worker joined
- Room name matches exactly
- Backend URL is correct in worker

### Issue: Wrong bot configuration
**Check:**
- `.env` has correct `TELNYX_DEFAULT_*` values
- Backend stores them in session
- Worker loads from session

---

## 📝 Summary

✅ **Backend and Worker are fully aligned for Telnyx phone calls**

- Room naming convention: ✅ Matched
- Session data structure: ✅ Matched
- Configuration defaults: ✅ Matched
- Livechat handling: ✅ Properly skipped for phone calls
- Bot configuration: ✅ Loaded from session

**Ready for testing!** 🚀

