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
        
        // Enable the join button by default
        const joinBtn = document.getElementById('joinBtn');
        if (joinBtn) {
            joinBtn.disabled = false;
            joinBtn.textContent = 'Join Voice Chat';
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
            // const response = await fetch('http://localhost:8000/api/connection_details', {
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
            
            this.showStatus(`
                🔗 Connected to Alive5 Support!<br>
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
                <span class="current-intent">Current Intent: <strong>${this.currentIntent || 'Not detected yet'}</strong></span>
                <span class="auto-detection">🤖 Auto-detection active</span>
            </div>
            <div class="log-entries" id="logEntries"></div>
        `;
        
        // Add a divider before the conversation log
        const divider = document.createElement('hr');
        divider.className = 'log-divider';
        
        // Add to container instead of controls for separate row layout
        const container = document.querySelector('.container');
        if (container) {
            container.appendChild(divider);
            container.appendChild(logContainer);
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
    
    // Toast notification for intent changes
    showToastNotification(message, type = 'info') {
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.style.position = 'fixed';
            toastContainer.style.top = '20px';
            toastContainer.style.right = '20px';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-icon">${type === 'intent-change' ? '🔄' : 'ℹ️'}</div>
            <div class="toast-content">${message}</div>
        `;
        
        // Style the toast
        toast.style.backgroundColor = type === 'intent-change' ? '#4a6da7' : '#444';
        toast.style.color = 'white';
        toast.style.padding = '12px 16px';
        toast.style.borderRadius = '4px';
        toast.style.marginBottom = '10px';
        toast.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
        toast.style.display = 'flex';
        toast.style.alignItems = 'center';
        toast.style.transition = 'all 0.3s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        
        // Add to container
        toastContainer.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        }, 10);
        
        // Auto-remove after delay
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(50px)';
            
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 5000);
    }
    
    updateDetectedIntent(newIntent) {
        if (newIntent !== this.currentIntent) {
            console.log(`INTENT_LOG: Intent changed from '${this.currentIntent || 'None'}' to '${newIntent}'`);
            const oldIntent = this.currentIntent;
            this.currentIntent = newIntent;
            
            // Update UI
            const intentStatus = document.querySelector('.current-intent');
            if (intentStatus) {
                intentStatus.innerHTML = `Current Intent: <strong>${newIntent}</strong> <span class="updated">✨ Updated</span>`;
            }
            
            // Show toast notification for intent changes
            this.showToastNotification(`Intent changed to: ${newIntent}`, 'intent-change');
            
            // Send update to server
            this.updateSessionIntent(newIntent);
        } else {
            console.log(`INTENT_LOG: Intent confirmed as '${newIntent}'`);
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
        // Handle room state changes
        this.room.on(LivekitClient.RoomEvent.Disconnected, () => {
            this.isConnected = false;
            this.showStatus('Disconnected from voice chat', 'info');
            document.getElementById('controls').classList.remove('show');
            document.getElementById('joinBtn').disabled = false;
            document.getElementById('joinBtn').textContent = 'Reconnect';
            
            // Hide transcript container
            document.getElementById('transcript-container').classList.remove('show');
        });
        
        this.room.on(LivekitClient.RoomEvent.Connected, () => {
            this.isConnected = true;
            this.showStatus('Connected to Alive5 Dynamic Voice Support', 'connected');
            document.getElementById('controls').classList.add('show');
            document.getElementById('joinBtn').disabled = true;
            
            // Show and initialize transcript container
            const transcriptContainer = document.getElementById('transcript-container');
            transcriptContainer.classList.add('show');
            document.getElementById('transcript-status').className = 'listening';
            document.getElementById('interim-transcript').textContent = '';
            document.getElementById('final-transcript').textContent = '';
            
            // Initialize transcript history
            this.interimTranscript = "";
            this.finalTranscript = "";
            
            // Send a test data message to ensure data channels are working
            setTimeout(() => {
                try {
                    // Check for different possible ways DataPacketKind might be exposed
                    let dataPacketKind;
                    
                    // Try different paths to find the RELIABLE constant
                    if (LivekitClient.DataPacketKind && LivekitClient.DataPacketKind.RELIABLE) {
                        dataPacketKind = LivekitClient.DataPacketKind.RELIABLE;
                    } else if (LivekitClient.DataPacketKind) {
                        // Some versions might use numeric values instead of named constants
                        dataPacketKind = 1; // 1 is typically RELIABLE
                    } else if (LivekitClient.Room && LivekitClient.Room.DataPacketKind) {
                        dataPacketKind = LivekitClient.Room.DataPacketKind.RELIABLE;
                    } else if (this.room.localParticipant && this.room.localParticipant.publishData) {
                        // If we can't find the constant, try without specifying kind (will use default)
                        this.room.localParticipant.publishData(
                            JSON.stringify({
                                type: 'client_info',
                                client: 'web',
                                version: '2.0',
                                timestamp: new Date().toISOString()
                            })
                        );
                        console.log('Sent client info data packet (default kind)');
                        return;
                    } else {
                        console.warn('Could not find DataPacketKind, skipping test packet');
                        return;
                    }
                    
                    this.room.localParticipant.publishData(
                        JSON.stringify({
                            type: 'client_info',
                            client: 'web',
                            version: '2.0',
                            timestamp: new Date().toISOString()
                        }),
                        dataPacketKind
                    );
                    console.log('Sent client info data packet');
                } catch (e) {
                    console.error('Failed to send test data packet:', e);
                }
            }, 2000);
        });
        
        // Handle remote audio tracks (Scott's voice)
        this.room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
            console.log('Track subscribed:', track.kind, 'from participant:', participant.identity);
            
            if (track.kind === 'audio') {
                // Create or get audio element for this participant
                let audioElement = document.getElementById(`audio-${participant.sid}`);
                if (!audioElement) {
                    audioElement = document.createElement('audio');
                    audioElement.id = `audio-${participant.sid}`;
                    audioElement.autoplay = true;
                    audioElement.controls = false; // Hide controls
                    document.body.appendChild(audioElement);
                    console.log('Created audio element for', participant.identity);
                }
                
                // Attach track to audio element
                track.attach(audioElement);
                console.log('Attached audio track to element');
                
                // Set volume to ensure it's audible
                audioElement.volume = 1.0;
                
                // Play audio - needed for some browsers
                try {
                    audioElement.play()
                        .then(() => console.log('Audio playback started'))
                        .catch(err => console.error('Audio playback failed:', err));
                } catch (e) {
                    console.warn('Audio play error:', e);
                }
            }
        });
        
        // Handle track unsubscribed
        this.room.on(LivekitClient.RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
            console.log('Track unsubscribed:', track.kind, 'from participant:', participant.identity);
            
            if (track.kind === 'audio') {
                // Remove audio element
                const audioElement = document.getElementById(`audio-${participant.sid}`);
                if (audioElement) {
                    track.detach(audioElement);
                    audioElement.remove();
                    console.log('Removed audio element for', participant.identity);
                }
            }
        });
        
        // Setup data channel for receiving transcripts
        this.room.on(LivekitClient.RoomEvent.DataReceived, (payload) => {
            try {
                console.log('Received data packet:', payload);
                let data;
                const self = this; // Preserve the 'this' context for callbacks
                
                // Direct handling for Uint8Array payload (common case from server)
                if (payload instanceof Uint8Array || payload instanceof ArrayBuffer) {
                    try {
                        const decoder = new TextDecoder();
                        const dataStr = decoder.decode(payload);
                        console.log('Directly decoded binary payload:', dataStr);
                        data = JSON.parse(dataStr);
                        console.log('Successfully parsed JSON from direct binary payload:', data);
                    } catch (directError) {
                        console.warn('Failed to directly parse binary payload:', directError);
                    }
                }
                
                // If direct parsing didn't work, try extracting from payload object
                if (!data) {
                    // Extract the actual data from the payload
                    let rawData = payload.data;
                    
                    // If payload.data is undefined, check if we can extract data from other payload properties
                    if (rawData === undefined) {
                        console.log('Data is undefined, checking payload properties:', payload);
                        if (payload.kind !== undefined && payload.payload) {
                            rawData = payload.payload;
                            console.log('Using payload.payload instead:', rawData);
                        } else if (typeof payload === 'string') {
                            rawData = payload;
                            console.log('Using payload directly as string:', rawData);
                        } else if (payload instanceof Uint8Array || payload instanceof ArrayBuffer) {
                            rawData = payload;
                            console.log('Using payload directly as binary data');
                        } else if (payload.toString && typeof payload.toString === 'function') {
                            // Last resort - try toString
                            rawData = payload.toString();
                            console.log('Using payload.toString():', rawData);
                        }
                    }
                    
                    // Now process rawData based on its type
                    if (rawData instanceof Uint8Array || rawData instanceof ArrayBuffer) {
                        // Convert binary data to string
                        const decoder = new TextDecoder();
                        const dataStr = decoder.decode(rawData);
                        console.log('Decoded binary data:', dataStr);
                        try {
                            data = JSON.parse(dataStr);
                            console.log('Parsed JSON from binary data:', data);
                        } catch (jsonError) {
                            console.warn('Received non-JSON data:', dataStr);
                            // If it's a simple string, treat it as transcript
                            if (dataStr.trim()) {
                                data = { transcript: dataStr, final: true };
                            } else {
                                console.log('Empty data string, skipping');
                            }
                        }
                    } else if (typeof rawData === 'string') {
                        try {
                            data = JSON.parse(rawData);
                            console.log('Parsed JSON from string data:', data);
                        } catch (jsonError) {
                            console.warn('Received non-JSON string:', rawData);
                            // If it's a simple string, treat it as transcript
                            if (rawData.trim()) {
                                data = { transcript: rawData, final: true };
                            } else {
                                console.log('Empty string data, skipping');
                            }
                        }
                    } else if (typeof rawData === 'object' && rawData !== null) {
                        // Already an object
                        data = rawData;
                        console.log('Using raw object data directly:', data);
                    } else if (rawData === undefined || rawData === null) {
                        console.warn('Empty data received, nothing to process');
                    } else {
                        console.warn('Unhandled data format:', typeof rawData, rawData);
                        // Try to convert to string as last resort
                        try {
                            const strData = String(rawData);
                            if (strData && strData.trim()) {
                                data = { transcript: strData, final: true };
                                console.log('Converted to string as fallback:', data);
                            } else {
                                console.log('Empty converted string, skipping');
                            }
                        } catch (conversionError) {
                            console.error('Failed to convert data to string:', conversionError);
                        }
                    }
                }
                
        console.log('Processed data:', data);
                
                // Handle transcript updates
                if (data && (data.type === 'transcript' || data.transcript || data.text)) {
                    try {
                        self.updateTranscript(data);
                        console.log('Transcript update processed successfully');
                    } catch (err) {
                        console.error('Error updating transcript:', err);
                    }
                }
                
                // Handle intent detection
                if (data && data.intent) {
                    try {
                        self.handleIntentUpdate(data.intent, data.intentSource || 'AI');
                        console.log('Intent update processed successfully');
                    } catch (err) {
                        console.error('Error handling intent update:', err);
                    }
                }
                
                // Handle user data extraction
                if (data && data.userData) {
                    try {
                        self.updateUserData(data.userData);
                        console.log('User data update processed successfully');
                    } catch (err) {
                        console.error('Error updating user data:', err);
                    }
                }
                
                // Handle system messages
                if (data && data.system) {
                    try {
                        self.addLogEntry({
                            speaker: 'System',
                            message: data.system,
                            type: 'system'
                        });
                        console.log('System message processed successfully:', data.system);
                        
                        // Show transcript container when we receive the initialization message
                        if (data.system.includes('Transcript capture initialized')) {
                            const container = document.getElementById('transcript-container');
                            if (container) {
                                container.classList.add('show');
                                console.log('Transcript container shown due to initialization message');
                            }
                        }
                    } catch (err) {
                        console.error('Error handling system message:', err);
                    }
                }
            } catch (e) {
                console.error('Error processing received data:', e);
            }
        });
    }
    
    // Add a new method to handle adding log entries
    addLogEntry(entry) {
        // Validate required fields
        if (!entry || !entry.message) {
            console.warn('Invalid log entry:', entry);
            return;
        }
        
        // Add timestamp if not provided
        if (!entry.timestamp) {
            entry.timestamp = new Date().toISOString();
        }
        
        // Store in conversation history
        this.conversationLog.push(entry);
        
        // Create or get the conversation log container
        let logContainer = document.querySelector('.conversation-log');
        if (!logContainer) {
            logContainer = document.createElement('div');
            logContainer.className = 'conversation-log';
            logContainer.innerHTML = `
                <h4>Conversation History</h4>
                <div class="intent-status">
                    <div class="current-intent">Current intent: <span id="current-intent">Detecting...</span></div>
                    <div class="auto-detection" id="intent-detection-status">Auto-detection active</div>
                </div>
                <div class="log-entries"></div>
            `;
            
            // Insert it after the transcript container
            const transcriptContainer = document.getElementById('transcript-container');
            if (transcriptContainer && transcriptContainer.parentNode) {
                transcriptContainer.parentNode.insertBefore(logContainer, transcriptContainer.nextSibling);
            } else {
                // Fallback - append to container
                document.querySelector('.container').appendChild(logContainer);
            }
        }
        
        // Get the log entries container
        const logEntries = logContainer.querySelector('.log-entries');
        if (!logEntries) {
            console.warn('Log entries container not found');
            return;
        }
        
        // Create the log entry element
        const logEntryEl = document.createElement('div');
        logEntryEl.className = `log-entry ${entry.type || ''}`;
        
        // Format timestamp
        let displayTime;
        try {
            const entryTime = new Date(entry.timestamp);
            displayTime = entryTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (e) {
            displayTime = 'now';
        }
        
        // Set the content
        logEntryEl.innerHTML = `
            <div class="log-header">
                <span class="speaker">${entry.speaker || 'Unknown'}</span>
                <span class="timestamp">${displayTime}</span>
            </div>
            <div class="log-message">${entry.message}</div>
        `;
        
        // Add intent change indicator if provided
        if (entry.intent) {
            const intentSpan = document.createElement('span');
            intentSpan.className = 'intent-change';
            intentSpan.textContent = `Intent: ${entry.intent}`;
            logEntryEl.querySelector('.log-header').appendChild(intentSpan);
        }
        
        // Add to the log
        logEntries.appendChild(logEntryEl);
        
        // Scroll to bottom
        logEntries.scrollTop = logEntries.scrollHeight;
        
        return logEntryEl;
    }
    
    // Handle intent updates
    handleIntentUpdate(intent, source) {
        if (!intent) return;
        
        // Update the current intent
        this.currentIntent = intent;
        
        // Update the UI
        const intentElement = document.getElementById('current-intent');
        if (intentElement) {
            const intentLabels = {
                sales: 'Sales & Pricing',
                support: 'Technical Support',
                billing: 'Billing & Account',
                general: 'General Inquiry'
            };
            
            intentElement.textContent = intentLabels[intent] || intent;
            intentElement.classList.add('updated');
            
            // Remove the updated class after animation
            setTimeout(() => {
                intentElement.classList.remove('updated');
            }, 2000);
        }
        
        // Update the detection status
        const detectionStatus = document.getElementById('intent-detection-status');
        if (detectionStatus) {
            detectionStatus.textContent = `Detected by ${source}`;
            detectionStatus.classList.add('updated');
            
            // Remove the updated class after animation
            setTimeout(() => {
                detectionStatus.classList.remove('updated');
            }, 2000);
        }
        
        // Log the intent change
        this.addLogEntry({
            speaker: 'System',
            message: `Intent changed to: ${intent}`,
            type: 'system',
            intent: intent
        });
        
        // Show a toast notification
        this.showToast(`Intent detected: ${intent}`, 'intent-change');
    }
    
    // Show a toast notification
    showToast(message, type = '') {
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">ℹ️</span>
            <span class="toast-message">${message}</span>
        `;
        
        // Add to container
        toastContainer.appendChild(toast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 3000);
    }
    
    // Update user data extracted from conversation
    updateUserData(userData) {
        if (!userData || typeof userData !== 'object') return;
        
        // Merge with existing data
        this.detectedUserData = { ...this.detectedUserData, ...userData };
        
        // Build a message with the extracted data
        const dataPoints = [];
        for (const [key, value] of Object.entries(userData)) {
            if (value) {
                // Format key for display (camelCase to Title Case)
                const formattedKey = key
                    .replace(/([A-Z])/g, ' $1')
                    .replace(/^./, str => str.toUpperCase());
                
                dataPoints.push(`<strong>${formattedKey}:</strong> ${value}`);
            }
        }
        
        if (dataPoints.length > 0) {
            // Add to conversation log
            this.addLogEntry({
                speaker: 'System',
                message: `User information detected:<br>${dataPoints.join('<br>')}`,
                type: 'system'
            });
            
            // Show toast notification
            this.showToast('User information updated', 'user-data');
        }
    }
    updateTranscript(data) {
        // Extract transcript information
        let transcriptText = '';
        let isFinal = false;
        
        // Log raw data to help with debugging
        console.log('UpdateTranscript called with data:', JSON.stringify(data));
        
        try {
            // Handle different transcript data formats
            if (data.transcript !== undefined) {
                if (typeof data.transcript === 'string') {
                    transcriptText = data.transcript;
                    isFinal = data.isFinal || data.final || false;
                    console.log('Using string transcript:', transcriptText);
                } else if (data.transcript && data.transcript.alternatives && data.transcript.alternatives.length > 0) {
                    transcriptText = data.transcript.alternatives[0].text;
                    isFinal = data.transcript.final || false;
                    console.log('Using structured transcript:', transcriptText);
                } else if (data.transcript === null || data.transcript === '') {
                    console.log('Empty transcript received, skipping');
                    return;
                }
            } else if (data.text !== undefined) {
                transcriptText = data.text;
                isFinal = data.isFinal || data.final || false;
                console.log('Using text field:', transcriptText);
            } else if (typeof data === 'string') {
                // Direct string transcript
                transcriptText = data;
                isFinal = true; // Assume final for direct strings
                console.log('Using direct string:', transcriptText);
            } else {
                console.warn('No recognized transcript format in data:', data);
                return;
            }
        } catch (error) {
            console.error('Error extracting transcript text:', error);
            return;
        }
        
        if (!transcriptText || transcriptText.trim() === '') {
            console.log('Empty transcript text after processing, skipping update');
            return;
        }
        
        console.log(`Transcript update: "${transcriptText}" (final: ${isFinal})`);
        
        try {
            // Update appropriate transcript based on final flag
            if (isFinal) {
                this.finalTranscript += (this.finalTranscript ? ' ' : '') + transcriptText;
                this.interimTranscript = '';
                
                // Log the final transcript as user message
                this.addLogEntry({
                    speaker: this.participantName || 'You',
                    message: transcriptText,
                    type: 'user'
                });
                console.log('Added final transcript to conversation log');
            } else {
                this.interimTranscript = transcriptText;
                console.log('Updated interim transcript');
            }
            
            // Update the UI
            const interimElement = document.getElementById('interim-transcript');
            const finalElement = document.getElementById('final-transcript');
            
            if (interimElement && finalElement) {
                interimElement.textContent = this.interimTranscript;
                finalElement.textContent = this.finalTranscript;
                
                // Update the timestamp
                this.lastTranscriptUpdate = Date.now();
                
                // Update transcript status
                const status = document.getElementById('transcript-status');
                if (status) {
                    if (this.interimTranscript) {
                        status.textContent = 'Transcribing...';
                        status.className = '';
                    } else {
                        status.textContent = 'Listening...';
                        status.className = 'listening';
                    }
                }
                
                // Make sure transcript container is visible
                const container = document.getElementById('transcript-container');
                if (container && !container.classList.contains('show')) {
                    container.classList.add('show');
                    console.log('Transcript container now visible');
                }
                
                console.log('Transcript UI updated successfully');
            } else {
                console.warn('Transcript UI elements not found');
            }
        } catch (uiError) {
            console.error('Error updating transcript UI:', uiError);
        }
    }
    
    handleDisconnection() {
        this.isConnected = false;
        this.hideControls();
        this.currentRoomName = null;
        
        // Cleanup audio elements more thoroughly
        try {
            // Find and remove all audio elements
            const audioElements = document.querySelectorAll('audio');
            console.log(`Cleaning up ${audioElements.length} audio elements`);
            
            audioElements.forEach(el => {
                try {
                    // Stop the audio playback first
                    if (el.srcObject) {
                        const tracks = el.srcObject.getTracks();
                        tracks.forEach(track => track.stop());
                    }
                    
                    // Clear the source and remove the element
                    el.srcObject = null;
                    el.remove();
                } catch (e) {
                    console.warn('Error cleaning up audio element:', e);
                }
            });
            
            // Cleanup any other media elements as well
            document.querySelectorAll('video').forEach(el => el.remove());
        } catch (e) {
            console.error('Error during media cleanup:', e);
        }
        
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