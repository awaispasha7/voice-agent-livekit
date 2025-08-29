class DynamicVoiceAgent {
    constructor() {
        this.room = null;
        this.isConnected = false;
        this.isMuted = false;
        this.localAudioTrack = null;
        this.currentRoomName = null;
        this.participantName = null;
        this.sessionId = this.generateSessionId();
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = 3;
        this.currentIntent = null;
        this.detectedUserData = {};
        this.conversationLog = [];
        
        // Transcript handling
        this.interimTranscript = "";
        this.finalTranscript = "";
        this.lastTranscriptUpdate = Date.now();
        
        this.bindEvents();
        this.updateDisplay();
    }
    
    generateSessionId() {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substring(2, 8);
        return `session_${timestamp}_${random}`;
    }
    
    updateDisplay() {
        const roomInput = document.getElementById('roomName');
        if (roomInput) {
            roomInput.style.display = 'none';
            const label = document.querySelector('label[for="roomName"]');
            if (label) label.style.display = 'none';
        }
        
        const statusEl = document.getElementById('status');
        if (statusEl) {
            statusEl.innerHTML = `
                <div class="session-info">
                    <strong>Session ID:</strong> <code>${this.sessionId}</code><br>
                    <strong>Status:</strong> Ready to connect to Alive5 Support<br>
                    <strong>Features:</strong> Dynamic Intent Detection, Real-time Analytics
                </div>
            `;
            statusEl.className = 'status info';
            statusEl.style.display = 'block';
        }
        
        const joinBtn = document.getElementById('joinBtn');
        if (joinBtn) {
            joinBtn.disabled = false;
            joinBtn.textContent = 'Join Voice Chat';
        }
    }
    
    bindEvents() {
        document.getElementById('joinForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.joinRoom();
        });
        
        document.getElementById('muteBtn').addEventListener('click', () => {
            this.toggleMute();
        });
        
        document.getElementById('disconnectBtn').addEventListener('click', () => {
            this.disconnect();
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'r' && e.ctrlKey && !this.isConnected) {
                e.preventDefault();
                this.reconnect();
            }
        });
        
        window.addEventListener('beforeunload', () => {
            if (this.isConnected) {
                this.disconnect(false);
            }
        });
    }
    
    async joinRoom() {
        this.participantName = document.getElementById('participantName').value.trim();
        
        if (!this.participantName) {
            this.showStatus('Please enter your name', 'error');
            return;
        }
        
        if (this.connectionAttempts >= this.maxConnectionAttempts) {
            this.showStatus('Maximum connection attempts reached. Please refresh the page.', 'error');
            return;
        }
        
        try {
            this.connectionAttempts++;
            this.showStatus('Connecting to Alive5 Support with dynamic intent detection...', 'connecting');
            document.getElementById('joinBtn').disabled = true;
            
            // Get connection details
            const response = await fetch('https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com/api/connection_details', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    participant_name: this.participantName,
                    user_data: {
                        session_start: new Date().toISOString()
                    }
                })
            });
            
            if (!response.ok) throw new Error('Failed to get connection details');
            const connectionDetails = await response.json();
            console.log('Connection details received:', connectionDetails);
            
            this.currentRoomName = connectionDetails.roomName;
            
            // Create room with proper configuration
            this.room = new LivekitClient.Room({
                adaptiveStream: true,
                dynacast: true,
                reconnectPolicy: {
                    nextRetryDelayInMs: (context) => {
                        console.log('Reconnection attempt:', context.retryCount);
                        return Math.min(1000 * Math.pow(2, context.retryCount), 30000);
                    },
                    maxRetries: 5,
                },
                publishDefaults: {
                    videoSimulcastLayers: [],
                    audioPreset: LivekitClient.AudioPresets.speech,
                }
            });
            
            this.setupRoomEvents();
            
            // Connect to room
            await this.room.connect(connectionDetails.serverUrl, connectionDetails.participantToken);
            await this.enableMicrophone();
            
            this.isConnected = true;
            this.connectionAttempts = 0;
            
            this.showStatus(`
                Connected to Alive5 Support!<br>
                Room: ${this.currentRoomName}<br>
                Scott is joining... (This may take 10-15 seconds)<br>
                Dynamic intent detection is active
            `, 'connected');
            
            this.showControls();
            this.initializeConversationLog();
            
        } catch (error) {
            console.error('Connection failed:', error);
            this.showStatus(`Connection failed: ${error.message}`, 'error');
            document.getElementById('joinBtn').disabled = false;
            
            if (this.room) {
                try {
                    await this.room.disconnect();
                } catch (e) {
                    console.warn('Error disconnecting after failed connection:', e);
                }
                this.room = null;
            }
        }
    }
    
    setupRoomEvents() {
        // Handle connection state
        this.room.on(LivekitClient.RoomEvent.Connected, () => {
            this.isConnected = true;
            this.showStatus('Connected to Alive5 Dynamic Voice Support', 'connected');
            document.getElementById('controls').classList.add('show');
            document.getElementById('joinBtn').disabled = true;
            
            this.createTranscriptContainer();
            console.log('Room connected - transcript container created');
        });
        
        this.room.on(LivekitClient.RoomEvent.Disconnected, () => {
            this.isConnected = false;
            this.handleDisconnection();
        });
        
        // Handle audio tracks from agent
        this.room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
            console.log('Track subscribed:', track.kind, 'from participant:', participant.identity);
            
            if (track.kind === 'audio') {
                let audioElement = document.getElementById(`audio-${participant.sid}`);
                if (!audioElement) {
                    audioElement = document.createElement('audio');
                    audioElement.id = `audio-${participant.sid}`;
                    audioElement.autoplay = true;
                    audioElement.controls = false;
                    document.body.appendChild(audioElement);
                    console.log('Created audio element for', participant.identity);
                }
                
                track.attach(audioElement);
                audioElement.volume = 1.0;
                
                try {
                    audioElement.play()
                        .then(() => console.log('Audio playback started'))
                        .catch(err => console.error('Audio playback failed:', err));
                } catch (e) {
                    console.warn('Audio play error:', e);
                }
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
            if (track.kind === 'audio') {
                const audioElement = document.getElementById(`audio-${participant.sid}`);
                if (audioElement) {
                    track.detach(audioElement);
                    audioElement.remove();
                }
            }
        });
        
        // CORRECT WAY: Register text stream handler for transcriptions
        this.room.registerTextStreamHandler('lk.transcription', async (reader, participantInfo) => {
            try {
                const message = await reader.readAll();
                console.log('Transcription received:', { message, participantInfo, attributes: reader.info.attributes });
                
                // Check if this is a user transcription (has transcribed_track_id attribute)
                if (reader.info.attributes && reader.info.attributes['lk.transcribed_track_id']) {
                    // This is a user's speech transcription
                    this.handleUserTranscription(message, participantInfo);
                } else {
                    // This is agent's speech transcription (synchronized with TTS)
                    this.handleAgentTranscription(message, participantInfo);
                }
            } catch (error) {
                console.error('Error processing transcription:', error);
            }
        });
        
        // Register handler for chat messages (text input)
        this.room.registerTextStreamHandler('lk.chat', async (reader, participantInfo) => {
            try {
                const message = await reader.readAll();
                console.log('Chat message received:', message, 'from:', participantInfo.identity);
                
                // Handle chat messages if needed
                this.addLogEntry({
                    speaker: participantInfo.identity,
                    message: message,
                    type: 'chat'
                });
            } catch (error) {
                console.error('Error processing chat message:', error);
            }
        });
    }
    
    handleUserTranscription(transcriptText, participantInfo) {
        console.log('User transcription:', transcriptText, 'from:', participantInfo.identity);
        
        // Update transcript display
        this.updateTranscriptDisplay(transcriptText, true); // true = final
        
        // Add to conversation log
        this.addLogEntry({
            speaker: this.participantName || 'You',
            message: transcriptText,
            type: 'user'
        });
        
        // Send transcript to backend for intent detection
        this.sendTranscriptForIntentDetection(transcriptText);
    }
    
    handleAgentTranscription(transcriptText, participantInfo) {
        console.log('Agent transcription:', transcriptText, 'from:', participantInfo.identity);
        
        // Add agent response to conversation log
        this.addLogEntry({
            speaker: participantInfo.identity === this.room.localParticipant.identity ? 'You' : 'Scott',
            message: transcriptText,
            type: 'agent'
        });
    }
    
    async sendTranscriptForIntentDetection(transcript) {
        try {
            // Send transcript to backend for intent processing
            const response = await fetch('https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com/api/process_transcript', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    room_name: this.currentRoomName,
                    transcript: transcript,
                    session_id: this.sessionId
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                if (result.intent && result.intent !== this.currentIntent) {
                    this.handleIntentUpdate(result.intent, 'AI');
                }
                if (result.userData) {
                    this.updateUserData(result.userData);
                }
            }
        } catch (error) {
            console.warn('Failed to send transcript for intent detection:', error);
        }
    }
    
    updateTranscriptDisplay(transcriptText, isFinal) {
        if (!transcriptText || transcriptText.trim() === '') return;
        
        const interimElement = document.getElementById('interim-transcript');
        const finalElement = document.getElementById('final-transcript');
        
        if (!interimElement || !finalElement) {
            console.warn('Transcript elements not found');
            this.createTranscriptContainer();
            return;
        }
        
        if (isFinal) {
            this.finalTranscript += (this.finalTranscript ? ' ' : '') + transcriptText;
            this.interimTranscript = '';
            
            finalElement.textContent = this.finalTranscript;
            interimElement.textContent = '';
        } else {
            this.interimTranscript = transcriptText;
            interimElement.textContent = this.interimTranscript;
        }
        
        // Update status
        const status = document.getElementById('transcript-status');
        if (status) {
            status.textContent = this.interimTranscript ? 'Transcribing...' : 'Listening...';
            status.className = this.interimTranscript ? 'transcribing' : 'listening';
        }
        
        this.lastTranscriptUpdate = Date.now();
    }
    
    createTranscriptContainer() {
        if (document.getElementById('transcript-container')) {
            return; // Already exists
        }
        
        const container = document.createElement('div');
        container.id = 'transcript-container';
        container.className = 'transcript-container show';
        container.style.display = 'block';
        
        container.innerHTML = `
            <div class="transcript-header">
                <span>🎤 Live Transcript</span>
                <span id="transcript-status" class="listening">Listening...</span>
            </div>
            <div id="transcript-body" class="transcript-body">
                <div id="interim-transcript" class="transcript-text interim"></div>
                <div id="final-transcript" class="transcript-text final"></div>
            </div>
        `;
        
        const controls = document.getElementById('controls');
        if (controls && controls.parentNode) {
            controls.parentNode.insertBefore(container, controls.nextSibling);
        } else {
            document.querySelector('.container').appendChild(container);
        }
        
        console.log('Transcript container created');
    }
    
    initializeConversationLog() {
        const logContainer = document.createElement('div');
        logContainer.id = 'conversationLog';
        logContainer.className = 'conversation-log';
        logContainer.innerHTML = `
            <h4>Conversation Log</h4>
            <div class="intent-status">
                <span class="current-intent">Current Intent: <strong id="current-intent-display">${this.currentIntent || 'Detecting...'}</strong></span>
                <span class="auto-detection">🤖 Auto-detection active</span>
            </div>
            <div class="log-entries" id="logEntries"></div>
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            const divider = document.createElement('hr');
            divider.className = 'log-divider';
            container.appendChild(divider);
            container.appendChild(logContainer);
        }
    }
    
    addLogEntry(entry) {
        if (!entry || !entry.message) {
            console.warn('Invalid log entry:', entry);
            return;
        }
        
        if (!entry.timestamp) {
            entry.timestamp = new Date().toISOString();
        }
        
        this.conversationLog.push(entry);
        
        let logContainer = document.querySelector('.conversation-log');
        if (!logContainer) {
            this.initializeConversationLog();
            logContainer = document.querySelector('.conversation-log');
        }
        
        const logEntries = logContainer.querySelector('.log-entries');
        if (!logEntries) return;
        
        const logEntryEl = document.createElement('div');
        logEntryEl.className = `log-entry ${entry.type || ''}`;
        
        let displayTime;
        try {
            const entryTime = new Date(entry.timestamp);
            displayTime = entryTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (e) {
            displayTime = 'now';
        }
        
        logEntryEl.innerHTML = `
            <div class="log-header">
                <span class="speaker">${entry.speaker || 'Unknown'}</span>
                <span class="timestamp">${displayTime}</span>
            </div>
            <div class="log-message">${entry.message}</div>
        `;
        
        if (entry.intent) {
            const intentSpan = document.createElement('span');
            intentSpan.className = 'intent-change';
            intentSpan.textContent = `Intent: ${entry.intent}`;
            logEntryEl.querySelector('.log-header').appendChild(intentSpan);
        }
        
        logEntries.appendChild(logEntryEl);
        logEntries.scrollTop = logEntries.scrollHeight;
        
        return logEntryEl;
    }
    
    handleIntentUpdate(intent, source) {
        if (!intent) return;
        
        const oldIntent = this.currentIntent;
        this.currentIntent = intent;
        
        const intentElement = document.getElementById('current-intent-display');
        if (intentElement) {
            const intentLabels = {
                sales: 'Sales & Pricing',
                support: 'Technical Support',
                billing: 'Billing & Account',
                general: 'General Inquiry'
            };
            
            intentElement.textContent = intentLabels[intent] || intent;
            intentElement.classList.add('updated');
            
            setTimeout(() => {
                intentElement.classList.remove('updated');
            }, 2000);
        }
        
        if (oldIntent !== intent) {
            this.addLogEntry({
                speaker: 'System',
                message: `Intent detected: ${intent} (by ${source})`,
                type: 'system',
                intent: intent
            });
            
            this.showToast(`Intent detected: ${intent}`, 'intent-change');
        }
        
        // Update session on backend
        this.updateSessionIntent(intent);
    }
    
    async updateSessionIntent(intent) {
        try {
            await fetch('https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com/api/sessions/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    room_name: this.currentRoomName,
                    intent: intent,
                    user_data: this.detectedUserData,
                    status: 'active'
                })
            });
        } catch (error) {
            console.warn('Failed to update session intent:', error);
        }
    }
    
    updateUserData(userData) {
        if (!userData || typeof userData !== 'object') return;
        
        this.detectedUserData = { ...this.detectedUserData, ...userData };
        
        const dataPoints = [];
        for (const [key, value] of Object.entries(userData)) {
            if (value) {
                const formattedKey = key
                    .replace(/([A-Z])/g, ' $1')
                    .replace(/^./, str => str.toUpperCase());
                
                dataPoints.push(`<strong>${formattedKey}:</strong> ${value}`);
            }
        }
        
        if (dataPoints.length > 0) {
            this.addLogEntry({
                speaker: 'System',
                message: `User information detected:<br>${dataPoints.join('<br>')}`,
                type: 'system'
            });
            
            this.showToast('User information detected', 'user-data');
        }
    }
    
    showToast(message, type = 'info') {
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
            `;
            document.body.appendChild(toastContainer);
        }
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const iconMap = {
            'intent-change': '🔄',
            'user-data': '👤',
            'info': 'ℹ️',
            'error': '❌'
        };
        
        toast.innerHTML = `
            <div class="toast-icon">${iconMap[type] || iconMap.info}</div>
            <div class="toast-content">${message}</div>
        `;
        
        toast.style.cssText = `
            background-color: ${type === 'intent-change' ? '#4a6da7' : type === 'error' ? '#d73027' : '#444'};
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            opacity: 0;
            transform: translateX(50px);
        `;
        
        toastContainer.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        }, 10);
        
        // Auto-remove
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(50px)';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }
    
    async enableMicrophone() {
        try {
            this.localAudioTrack = await LivekitClient.createLocalAudioTrack({
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000,
            });
            
            await this.room.localParticipant.publishTrack(this.localAudioTrack);
            console.log('Microphone enabled with speech optimization');
        } catch (error) {
            console.error('Failed to enable microphone:', error);
            throw new Error('Microphone access denied. Please allow microphone access and try again.');
        }
    }
    
    toggleMute() {
        if (!this.localAudioTrack || !this.isConnected) return;
        
        this.isMuted = !this.isMuted;
        this.localAudioTrack.mute(this.isMuted);
        
        const btn = document.getElementById('muteBtn');
        btn.textContent = this.isMuted ? '🔊 Unmute' : '🔇 Mute';
        btn.classList.toggle('muted', this.isMuted);
        
        this.showStatus(
            this.isMuted ? 'Microphone muted' : 'Microphone unmuted', 
            this.isMuted ? 'warning' : 'connected'
        );
        
        this.addLogEntry({
            speaker: 'System',
            message: this.isMuted ? 'Microphone muted' : 'Microphone unmuted',
            type: 'system'
        });
    }
    
    async sendTextMessage(text) {
        if (!this.room || !this.isConnected) return;
        
        try {
            await this.room.localParticipant.sendText(text, {
                topic: 'lk.chat'
            });
            
            this.addLogEntry({
                speaker: 'You',
                message: text,
                type: 'text-input'
            });
        } catch (error) {
            console.error('Failed to send text message:', error);
        }
    }
    
    async disconnect(showMessage = true) {
        if (!this.room) return;
        
        try {
            if (this.isConnected) {
                await this.room.disconnect();
                console.log('Manually disconnected from room:', this.currentRoomName);
                
                if (this.currentRoomName) {
                    fetch(`https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com/api/rooms/${this.currentRoomName}`, {
                        method: 'DELETE'
                    }).catch(e => console.warn('Session cleanup notification failed:', e));
                }
            }
        } catch (error) {
            console.error('Error during disconnect:', error);
        } finally {
            this.room = null;
            if (showMessage) {
                this.handleDisconnection();
            }
        }
    }
    
    handleDisconnection() {
        this.isConnected = false;
        this.hideControls();
        this.currentRoomName = null;
        
        // Cleanup audio elements
        document.querySelectorAll('audio').forEach(el => {
            try {
                if (el.srcObject) {
                    const tracks = el.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                }
                el.srcObject = null;
                el.remove();
            } catch (e) {
                console.warn('Error cleaning up audio element:', e);
            }
        });
        
        document.querySelectorAll('video').forEach(el => el.remove());
        
        const summary = this.generateSessionSummary();
        this.showStatus(`
            Session Ended<br>
            ${summary}<br>
            Click "Join Voice Chat" to start a new session.
        `, 'info');
        
        document.getElementById('joinBtn').disabled = false;
        document.getElementById('joinBtn').textContent = 'Join Voice Chat';
        
        // Reset for next session
        this.currentIntent = null;
        this.detectedUserData = {};
        this.conversationLog = [];
        this.sessionId = this.generateSessionId();
        this.updateDisplay();
    }
    
    generateSessionSummary() {
        const duration = this.conversationLog.length > 0 ? 
            `Duration: ${Math.ceil(this.conversationLog.length / 2)} exchanges` : 
            'Brief session';
        
        const finalIntent = this.currentIntent || 'Unknown';
        const dataCollected = Object.keys(this.detectedUserData).length > 0 ? 
            'User data collected' : 
            'No data collected';
        
        return `Final Intent: ${finalIntent} | ${duration} | ${dataCollected}`;
    }
    
    async reconnect() {
        if (!this.isConnected) {
            this.connectionAttempts = 0;
            await this.joinRoom();
        }
    }
    
    showStatus(message, type) {
        const el = document.getElementById('status');
        el.innerHTML = message;
        el.className = `status ${type}`;
        el.style.display = 'block';
    }
    
    showControls() {
        document.getElementById('controls').classList.add('show');
    }
    
    hideControls() {
        document.getElementById('controls').classList.remove('show');
        
        const muteBtn = document.getElementById('muteBtn');
        muteBtn.textContent = '🔇 Mute';
        muteBtn.classList.remove('muted');
        this.isMuted = false;
        
        const logContainer = document.getElementById('conversationLog');
        if (logContainer) {
            logContainer.remove();
        }
        
        const transcriptContainer = document.getElementById('transcript-container');
        if (transcriptContainer) {
            transcriptContainer.classList.remove('show');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DynamicVoiceAgent();
});