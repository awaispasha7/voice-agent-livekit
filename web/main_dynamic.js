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
        
        this.bindEvents();
        this.updateDisplay();
        this.initializeIntentButtons();
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
    }
    
    initializeIntentButtons() {
        // Create intent selection buttons
        const intentContainer = document.createElement('div');
        intentContainer.id = 'intentSelection';
        intentContainer.className = 'intent-selection';
        intentContainer.innerHTML = `
            <h3>What can we help you with today?</h3>
            <div class="intent-buttons">
                <button class="intent-btn" data-intent="sales">💰 Sales & Pricing</button>
                <button class="intent-btn" data-intent="support">🔧 Technical Support</button>
                <button class="intent-btn" data-intent="billing">💳 Billing & Account</button>
                <button class="intent-btn" data-intent="general">💬 General Inquiry</button>
            </div>
            <p class="intent-help">Select your topic or just start talking - our AI will detect your intent automatically!</p>
        `;
        
        // Insert before the form
        const form = document.getElementById('joinForm');
        if (form) {
            form.parentNode.insertBefore(intentContainer, form);
        }
        
        // Add event listeners for intent buttons
        document.querySelectorAll('.intent-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.selectIntent(btn.dataset.intent);
            });
        });
    }
    
    selectIntent(intent) {
        this.currentIntent = intent;
        
        // Update UI to show selected intent
        document.querySelectorAll('.intent-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        document.querySelector(`[data-intent="${intent}"]`).classList.add('selected');
        
        // Update status
        const intentLabels = {
            sales: 'Sales & Pricing Inquiry',
            support: 'Technical Support Request',
            billing: 'Billing & Account Question',
            general: 'General Inquiry'
        };
        
        this.showStatus(`Selected: ${intentLabels[intent]}. Ready to connect!`, 'info');
        
        // Enable the join button
        document.getElementById('joinBtn').disabled = false;
        document.getElementById('joinBtn').textContent = `Connect for ${intentLabels[intent]}`;
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
        
        // Add intent switching during conversation
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
        
        if (!this.currentIntent) {
            this.showStatus('Please select what you need help with', 'error');
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
            
            // Get connection details with intent information
            // const response = await fetch('http://localhost:8000/api/connection_details', {
            const response = await fetch('https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com/api/connection_details', {
            method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    participant_name: this.participantName,
                    intent: this.currentIntent,
                    user_data: {
                        initial_intent: this.currentIntent,
                        session_start: new Date().toISOString()
                    }
                })
            });
            
            if (!response.ok) throw new Error('Failed to get connection details');
            const connectionDetails = await response.json();
            console.log('Enhanced connection details:', connectionDetails);
            
            this.currentRoomName = connectionDetails.roomName;
            
            // Create room with enhanced settings
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
                // Enable data channels for intent updates
                publishDefaults: {
                    videoSimulcastLayers: [],
                    audioPreset: LivekitClient.AudioPresets.speech,
                }
            });
            
            this.setupEnhancedRoomEvents();
            
            // Connect to room
            await this.room.connect(connectionDetails.serverUrl, connectionDetails.participantToken);
            await this.enableMicrophone();
            
            this.isConnected = true;
            this.connectionAttempts = 0;
            
            const intentLabels = {
                sales: 'Sales Team',
                support: 'Technical Support',
                billing: 'Billing Department',
                general: 'General Support'
            };
            
            this.showStatus(`
                🔗 Connected to Alive5 ${intentLabels[this.currentIntent]}!<br>
                Room: ${this.currentRoomName}<br>
                ⏳ Scott is joining... (This may take 10-15 seconds)<br>
                🤖 Dynamic intent detection is active
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
    
    initializeConversationLog() {
        // Create conversation log UI
        const logContainer = document.createElement('div');
        logContainer.id = 'conversationLog';
        logContainer.className = 'conversation-log';
        logContainer.innerHTML = `
            <h4>Conversation Log</h4>
            <div class="intent-status">
                <span class="current-intent">Current Intent: <strong>${this.currentIntent}</strong></span>
                <span class="auto-detection">🤖 Auto-detection active</span>
            </div>
            <div class="log-entries" id="logEntries"></div>
        `;
        
        const controls = document.getElementById('controls');
        if (controls) {
            controls.appendChild(logContainer);
        }
    }
    
    addToConversationLog(speaker, message, intent = null) {
        const logEntries = document.getElementById('logEntries');
        if (!logEntries) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `log-entry ${speaker.toLowerCase()}`;
        
        let intentBadge = '';
        if (intent && intent !== this.currentIntent) {
            intentBadge = `<span class="intent-change">Intent detected: ${intent}</span>`;
            this.updateDetectedIntent(intent);
        }
        
        entry.innerHTML = `
            <div class="log-header">
                <span class="speaker">${speaker}:</span>
                <span class="timestamp">${timestamp}</span>
                ${intentBadge}
            </div>
            <div class="log-message">${message}</div>
        `;
        
        logEntries.appendChild(entry);
        logEntries.scrollTop = logEntries.scrollHeight;
        
        // Store in conversation log
        this.conversationLog.push({
            speaker,
            message,
            timestamp,
            intent
        });
    }
    
    updateDetectedIntent(newIntent) {
        if (newIntent !== this.currentIntent) {
            this.currentIntent = newIntent;
            console.log("New Intent detected:", newIntent);
            
            // Update UI
            const intentStatus = document.querySelector('.current-intent');
            if (intentStatus) {
                intentStatus.innerHTML = `Current Intent: <strong>${newIntent}</strong> <span class="updated">✨ Updated</span>`;
            }
            
            // Send update to server
            this.updateSessionIntent(newIntent);
            
            console.log(`Intent updated to: ${newIntent}`);
        }
    }
    
    async updateSessionIntent(intent) {
        try {
            // await fetch('http://localhost:8000/api/sessions/update', {
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
    
    setupEnhancedRoomEvents() {
        // Handle incoming audio (agent speaking)
        this.room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
            console.log('Track subscribed:', track.kind, 'from', participant.identity);
            if (participant.sid === this.room.localParticipant.sid) return;
            
            if (track.kind === LivekitClient.Track.Kind.Audio) {
                console.log('Setting up agent audio...');
                const audioElement = track.attach();
                audioElement.style.display = 'none';
                audioElement.volume = 1.0;
                document.body.appendChild(audioElement);
                
                const playAudio = async () => {
                    try {
                        await audioElement.play();
                        console.log('Agent audio is playing successfully');
                    } catch (e) {
                        console.error('Audio play failed:', e);
                        if (e.name === 'NotAllowedError') {
                            this.showStatus('⚠️ Please click anywhere to enable audio playback', 'warning');
                            document.addEventListener('click', async () => {
                                try {
                                    await audioElement.play();
                                    console.log('Audio started after user interaction');
                                } catch (err) {
                                    console.warn('Audio still failed after interaction:', err);
                                }
                            }, { once: true });
                        }
                    }
                };
                
                playAudio();
            }
        });
        
        // Handle data messages (for intent updates from agent)
        this.room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant) => {
            try {
                const data = JSON.parse(new TextDecoder().decode(payload));
                console.log('Received data from agent:', data);
                
                if (data.type === 'intent_detection') {
                    this.updateDetectedIntent(data.intent);
                    this.addToConversationLog('System', `Intent detected: ${data.intent}`, data.intent);
                }
                
                if (data.type === 'user_data_extracted') {
                    this.detectedUserData = { ...this.detectedUserData, ...data.data };
                    console.log('User data extracted:', this.detectedUserData);
                }
                
                if (data.type === 'transcript') {
                    this.addToConversationLog(data.speaker, data.message);
                }
            } catch (e) {
                console.warn('Failed to parse data message:', e);
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.Connected, () => {
            console.log('✅ Connected to dynamic room:', this.currentRoomName);
            console.log('Local participant:', this.room.localParticipant.identity);
        });
        
        this.room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
            console.log('❌ Disconnected from room. Reason:', reason);
            this.handleDisconnection();
        });
        
        this.room.on(LivekitClient.RoomEvent.ParticipantConnected, (participant) => {
            console.log('Participant connected:', participant.identity);
            if (participant.identity.toLowerCase().includes('agent') || 
                participant.identity.toLowerCase().includes('scott')) {
                this.showStatus('🎤 Scott has joined with dynamic AI capabilities! Start talking - he can detect your intent automatically!', 'connected');
                this.addToConversationLog('System', 'Scott (AI Agent) has joined the conversation');
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.ParticipantDisconnected, (participant) => {
            console.log('❌ Participant disconnected:', participant.identity);
            if (participant.identity.toLowerCase().includes('agent') || 
                participant.identity.toLowerCase().includes('scott')) {
                this.showStatus('⚠️ Scott has left the conversation. The session may have ended.', 'warning');
                this.addToConversationLog('System', 'Scott has left the conversation');
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.ConnectionQualityChanged, (quality, participant) => {
            if (participant === this.room.localParticipant) {
                console.log('Connection quality:', quality);
                if (quality === 'poor') {
                    this.showStatus('Connection quality is poor. Audio may be affected.', 'warning');
                }
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.Reconnecting, () => {
            this.showStatus('Connection lost. Reconnecting...', 'connecting');
        });
        
        this.room.on(LivekitClient.RoomEvent.Reconnected, () => {
            this.showStatus('Reconnected successfully! Dynamic features restored.', 'connected');
        });
    }
    
    handleDisconnection() {
        this.isConnected = false;
        this.hideControls();
        this.currentRoomName = null;
        
        // Cleanup audio elements
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(el => el.remove());
        
        // Show session summary
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
    
    async enableMicrophone() {
        try {
            this.localAudioTrack = await LivekitClient.createLocalAudioTrack({
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000, // Optimal for speech recognition
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
        
        this.addToConversationLog('System', this.isMuted ? 'Microphone muted' : 'Microphone unmuted');
    }
    
    async requestTransfer(department) {
        try {
            // const response = await fetch(`http://localhost:8000/api/sessions/${this.currentRoomName}/transfer`, {
            const response = await fetch(`https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com/api/sessions/${this.currentRoomName}/transfer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ department })
            });
            
            if (response.ok) {
                this.showStatus(`Transfer to ${department} initiated. Please hold...`, 'info');
                this.addToConversationLog('System', `Transfer to ${department} requested`);
            }
        } catch (error) {
            console.error('Transfer request failed:', error);
        }
    }
    
    async disconnect(showMessage = true) {
        if (!this.room) return;
        
        try {
            if (this.isConnected) {
                await this.room.disconnect();
                console.log('Manually disconnected from room:', this.currentRoomName);
                
                // Notify backend about session completion
                if (this.currentRoomName) {
                    // fetch(`http://localhost:8000/api/rooms/${this.currentRoomName}`, {
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
        
        // Remove conversation log
        const logContainer = document.getElementById('conversationLog');
        if (logContainer) {
            logContainer.remove();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DynamicVoiceAgent();
});