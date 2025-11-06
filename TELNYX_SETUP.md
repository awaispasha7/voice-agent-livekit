# Telnyx + LiveKit SIP Integration Setup Guide

## Overview
This guide explains how to set up Telnyx phone calls with LiveKit voice agent.

## Architecture Flow

```
Phone Call → Telnyx → LiveKit SIP Domain → LiveKit Room → Voice Agent
```

## Step 1: Configure LiveKit SIP Trunk (Inbound)

1. **Go to LiveKit Cloud Dashboard**
   - Navigate to: https://cloud.livekit.io
   - Go to **SIP** → **Trunks** → **Create Trunk**

2. **Configure Inbound Trunk**
   - **Name**: `telnyx-inbound`
   - **Type**: Inbound
   - **Allowed Number Pattern**: `+*` (allows any number)
   - **Room Name Pattern**: `telnyx_call_{call_id}` or custom pattern
   - **Note the Trunk ID** - you'll need this for `LIVEKIT_SIP_TRUNK_ID`

3. **Get SIP Domain**
   - Your SIP domain is: `s95527eskik.sip.livekit.cloud`
   - This is already in your `.env` as `LIVEKIT_SIP_DOMAIN`

## Step 2: Configure Telnyx SIP Connection

1. **Go to Telnyx Dashboard**
   - Navigate to: https://portal.telnyx.com
   - Go to **Voice** → **SIP Connections**

2. **Edit Your SIP Connection** (ID: `2756143467015963992`)
   - **Outbound Settings**:
     - **Outbound Voice Profile**: Select your voice profile
     - **SIP URI**: `sip:s95527eskik.sip.livekit.cloud`
     - **Transport Protocol**: UDP (or TCP)
   
3. **Configure Webhook**
   - **Webhook URL**: `https://18.210.238.67.nip.io/api/telnyx/webhook`
   - **Webhook Event Filters**: Select:
     - `call.initiated`
     - `call.hangup`
     - `call.answered`

## Step 3: Configure Phone Number

1. **In Telnyx Dashboard**
   - Go to **Numbers** → **Your Numbers**
   - Find number: `+14153765236`
   - **Connection**: Select your SIP Connection (`2756143467015963992`)
   - **Webhook URL**: Same as above (`/api/telnyx/webhook`)

## Step 4: Update Environment Variables

Add to your `.env` file:
```bash
# Get this from LiveKit dashboard after creating trunk
LIVEKIT_SIP_TRUNK_ID=your_trunk_id_here

# Default bot configuration for phone calls
TELNYX_DEFAULT_BOTCHAIN=voice-1
TELNYX_DEFAULT_ORG=alive5stage0
TELNYX_DEFAULT_FAQ_BOT=faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e
```

## Step 5: How It Works

### Call Flow:
1. **Incoming Call**: Someone calls `+14153765236`
2. **Telnyx Webhook**: Sends `call.initiated` event to `/api/telnyx/webhook`
3. **Backend Creates Room**: Creates LiveKit room with name `telnyx_call_{call_control_id}`
4. **Backend Answers Call**: Returns Telnyx answer command
5. **LiveKit Connects**: SIP trunk automatically connects call to room
6. **Worker Joins**: Your voice agent worker automatically joins the room
7. **Conversation Starts**: Agent greets caller and handles conversation

### Call End Flow:
1. **Call Hangs Up**: Telnyx sends `call.hangup` event
2. **Backend Cleanup**: Deletes LiveKit room
3. **Session Cleaned**: Removes session data

## Step 6: Testing

1. **Call Your Number**: `+14153765236`
2. **Check Backend Logs**: Should see:
   ```
   📞 Telnyx webhook received: call.initiated
   📞 Incoming call from +1234567890 to +14153765236
   ✅ Created LiveKit room: telnyx_call_xxx
   ```
3. **Check Worker Logs**: Should see agent joining room
4. **Verify Agent Responds**: Agent should greet caller

## Troubleshooting

### Call Not Connecting
- Verify `LIVEKIT_SIP_TRUNK_ID` is set correctly
- Check Telnyx SIP Connection is configured correctly
- Verify webhook URL is accessible from internet

### Agent Not Joining
- Check worker is running and connected to LiveKit
- Verify room name pattern matches what Telnyx creates
- Check worker logs for errors

### No Audio
- Verify SIP trunk is configured for inbound
- Check Telnyx voice profile settings
- Verify codecs are compatible

## API Endpoints

- `POST /api/telnyx/webhook` - Handles Telnyx call events
- `GET /api/telnyx/sip/trunk/setup` - Returns setup instructions

## Notes

- Each call creates a unique room: `telnyx_call_{call_control_id}`
- Rooms auto-cleanup after 5 minutes of inactivity
- Default bot configuration is used for all phone calls
- Caller number is stored in session metadata

