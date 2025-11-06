# 855 Number Setup Guide

## Overview

Your main customer-facing number is **+1 (855) 551-8858**. Currently, it forwards to a call center (Posh). To use it with your AI voice agent, you have two options.

---

## Option 1: Forward from Current Provider (Quick Setup)

### Steps:

1. **Log into your current phone provider** (whoever manages 855-551-8858)
2. **Configure Call Forwarding:**
   - Forward all calls from `855-551-8858` to `+14153765236` (your Telnyx number)
   - Set forwarding type: "Always Forward" or "Forward All Calls"

3. **Test:**
   - Call 855-551-8858
   - Should forward to 415-376-5236
   - AI agent should answer

### Flow:
```
Caller → 855-551-8858 → Current Provider → Forwards to +14153765236 → Telnyx → AI Agent
```

### Pros:
- ✅ Quick setup (5 minutes)
- ✅ No number porting needed
- ✅ Can revert easily

### Cons:
- ❌ Depends on your provider's forwarding settings
- ❌ May have forwarding delays
- ❌ Caller ID might show as forwarded number

---

## Option 2: Port Number to Telnyx (Recommended)

### Steps:

1. **Initiate Number Port in Telnyx:**
   - Go to: Telnyx Dashboard → Numbers → Port Numbers
   - Click "Port a Number"
   - Enter: `+18555518858`
   - Provide required documentation (LOA - Letter of Authorization)
   - Submit port request

2. **Wait for Port Completion:**
   - Porting typically takes 1-2 weeks
   - Telnyx will notify you when complete

3. **Assign to Bridge App:**
   - Once ported, go to: Numbers → My Numbers
   - Find `+18555518858`
   - Set Connection/Application: "Alive5 Bridge App"
   - Save

4. **Test:**
   - Call 855-551-8858
   - Should go directly to Telnyx → AI Agent

### Flow:
```
Caller → 855-551-8858 → Telnyx (direct) → AI Agent
```

### Pros:
- ✅ Full control over the number
- ✅ No forwarding delays
- ✅ Better call quality
- ✅ Same setup as 415 number
- ✅ Can use for transfers (if needed)

### Cons:
- ❌ Takes 1-2 weeks to port
- ❌ Requires documentation
- ❌ Temporary downtime during port

---

## Recommendation

**For immediate testing:** Use **Option 1** (forwarding) to test the AI agent flow.

**For production:** Use **Option 2** (port to Telnyx) for better control and reliability.

---

## Current Configuration

Your `.env` already has:
```bash
TELNYX_CALL_CENTER_NUMBER=+18555518858
```

This is used for:
- **Call transfers**: When AI needs to transfer to a human agent
- **Future use**: If you want to transfer calls back to the 855 number

**Note:** If you port the 855 number to Telnyx, you can still use it for transfers, or keep it as the main number and use a different number for transfers.

---

## Testing

Once configured (either option), test by:

1. **Call 855-551-8858**
2. **Monitor logs:**
   ```powershell
   ./logs-backend.ps1
   ./logs-worker.ps1
   ```
3. **Expected flow:**
   - Call connects
   - AI agent greets you
   - Conversation begins

---

## Questions?

- **Which provider manages 855-551-8858?** (needed for Option 1)
- **Do you want to port it?** (needed for Option 2)
- **Do you need both numbers active?** (855 for customers, 415 for testing)

