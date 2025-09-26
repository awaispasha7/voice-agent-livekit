class DynamicVoiceAgent {
    constructor() {
        this.room = null;
        this.isConnected = false;
        this.localAudioTrack = null;
        this.currentRoomName = null;
        this.participantName = null;
        this.sessionId = this.generateSessionId();
        this.connectionAttempts = 0;
        this.maxConnectionAttempts = 3;
        this.currentIntent = null;
        this.detectedUserData = {};
        this.conversationLog = [];
        this.selectedVoice = 'a0e99841-438c-4a64-b679-ae501e7d6091'; // Default voice (Sonic)
        
        // Track processed messages to prevent duplicates
        this.processedMessages = new Set();
        this.lastProcessedMessage = null;
        this.lastProcessedTime = 0;
        this.transcriptTimeout = null;
        this.pendingTranscript = null;
        
        // Track conversation history for intent detection
        this.conversationHistory = [];
        
        // Transcript handling
        this.interimTranscript = "";
        this.finalTranscript = "";
        this.lastTranscriptUpdate = Date.now();
        
        // Configuration
        this.config = {
            API_BASE_URL: window.API_BASE_URL || window.BACKEND_URL || 'http://localhost:8000',
            ENDPOINTS: {
                CONNECTION_DETAILS: '/api/connection_details',
                PROCESS_FLOW_MESSAGE: '/api/process_flow_message',
                UPDATE_SESSION: '/api/sessions/update',
                DELETE_ROOM: '/api/rooms'
            },
            CONNECTION: {
                MAX_ATTEMPTS: 3,
                RETRY_DELAY: 2000,
                TIMEOUT: 10000
            }
        };
        
        this.bindEvents();
        this.updateDisplay();
        
        // Initialize intent display to show "Offline" by default
        this.updateIntentDisplay();
    }
    
    generateSessionId() {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substring(2, 8);
        return `session_${timestamp}_${random}`;
    }
    
    // Helper function to try multiple URLs if one fails
    async fetchWithFallback(endpoint, options = {}) {
        const urls = [
            this.config.API_BASE_URL
        ];
        
        for (const baseUrl of urls) {
            try {
                const url = baseUrl + endpoint;
                
                const response = await fetch(url, {
                    ...options,
                    timeout: this.config.CONNECTION.TIMEOUT
                });
                
                if (response.ok) {
                    return response;
                }
            } catch (error) {
                // Continue to next URL
            }
        }
        
        throw new Error(`All API endpoints failed for ${endpoint}`);
    }
    
    hashMessage(message) {
        // Simple hash function for message deduplication
        let hash = 0;
        const str = message.toLowerCase().trim();
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString();
    }
    
    
    processCompleteTranscript(transcriptText) {
        // Prevent duplicate processing of the same message
        const messageHash = this.hashMessage(transcriptText);
        const currentTime = Date.now();
        
        // Check if we've already processed this exact message recently (within 5 seconds)
        if (this.processedMessages.has(messageHash) || 
            (this.lastProcessedMessage === transcriptText && (currentTime - this.lastProcessedTime) < 5000)) {
            // Skipping duplicate message
            return;
        }
        
        // Also check if this is a partial version of a message we already processed
        if (this.lastProcessedMessage && 
            this.lastProcessedMessage.includes(transcriptText) && 
            (currentTime - this.lastProcessedTime) < 5000) {
            // Skipping partial message
            return;
        }
        
        // Also check if this transcript is a substring of a message we already processed
        if (this.lastProcessedMessage && 
            transcriptText.length < this.lastProcessedMessage.length &&
            this.lastProcessedMessage.startsWith(transcriptText) &&
            (currentTime - this.lastProcessedTime) < 5000) {
            // Skipping substring message
            return;
        }
        
        // Mark this message as processed
        this.processedMessages.add(messageHash);
        this.lastProcessedMessage = transcriptText;
        this.lastProcessedTime = currentTime;
        
        // Clean up old processed messages (keep only last 10)
        if (this.processedMessages.size > 10) {
            const messagesArray = Array.from(this.processedMessages);
            this.processedMessages.clear();
            // Keep the last 5 messages
            messagesArray.slice(-5).forEach(msg => this.processedMessages.add(msg));
        }
        
        // Check if this looks like a farewell
        const isFarewell = this.detectFarewell(transcriptText);
        
        // Add to conversation log with clear user identification
        this.addLogEntry({
            speaker: `👤 ${this.participantName || 'You'}`,
            message: transcriptText,
            type: 'user'
        });
        
        // Add to conversation history for intent detection
        this.conversationHistory.push({
            role: 'user',
            content: transcriptText,
            timestamp: new Date().toISOString()
        });
        
        // Send for flow processing
        this.sendTranscriptForFlowProcessing(transcriptText);
    }
    
    updateDisplay() {
        const roomInput = document.getElementById('roomName');
        if (roomInput) {
            roomInput.style.display = 'none';
            const label = document.querySelector('label[for="roomName"]');
            if (label) label.style.display = 'none';
        }
        
        const sessionDetails = document.getElementById('sessionDetails');
        const sessionIdDisplay = document.getElementById('sessionIdDisplay');
        if (sessionDetails && sessionIdDisplay) {
            sessionIdDisplay.textContent = this.sessionId;
            sessionDetails.style.display = 'block';
        }
        
        const joinBtn = document.getElementById('joinBtn');
        if (joinBtn) {
            joinBtn.disabled = false;
            joinBtn.textContent = 'Join Voice Chat';
        }
    }
    
    bindEvents() {
        const joinForm = document.getElementById('joinForm');
        if (joinForm) {
            joinForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.joinRoom();
        });
        }
        
        const disconnectBtn = document.getElementById('disconnectBtn');
        if (disconnectBtn) {
            disconnectBtn.addEventListener('click', () => {
                this.disconnect();
            });
        }

        // Voice selection handlers
        const voiceSelect = document.getElementById('voiceSelect');
        if (voiceSelect) {
            voiceSelect.addEventListener('change', (e) => {
                this.selectedVoice = e.target.value;
            });
        }

        const voiceChangeSelect = document.getElementById('voiceChangeSelect');
        if (voiceChangeSelect) {
            voiceChangeSelect.addEventListener('change', (e) => {
                this.changeVoice(e.target.value);
            });
        }
        
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
        // Generate unique anonymous participant name
        this.participantName = `User_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`;
        
        // Get botchain and org info from form
        const botchainName = document.getElementById('botchainName')?.value?.trim() || null;
        const orgName = document.getElementById('orgName')?.value?.trim() || null;
        const selectedVoice = document.getElementById('voiceSelect')?.value || this.selectedVoice;
        
        if (!botchainName) {
            this.showStatus('Please enter a bot name (botchain)', 'error');
            return;
        }
        
        // Reset connection attempts for new manual attempts
        this.connectionAttempts = 0;
        
        try {
            this.connectionAttempts++;
            this.showStatus('Loading bot configuration...', 'connecting');
            document.getElementById('joinBtn').disabled = true;
            
            // Step 1: Validate and load template
            const templateResponse = await this.fetchWithFallback('/api/validate_and_load_template', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    botchain_name: botchainName,
                    org_name: orgName || 'alive5stage0'
                })
            });
            
            if (!templateResponse.ok) {
                throw new Error('Failed to validate bot configuration');
            }
            
            const templateResult = await templateResponse.json();
            
            if (templateResult.status !== 'success') {
                // Handle different error types
                let errorMessage = templateResult.message;
                if (templateResult.error_type === 'not_found') {
                    errorMessage = `Bot "${botchainName}" not found. Please check the bot name and try again.`;
                } else if (templateResult.error_type === 'timeout') {
                    errorMessage = `Timeout loading bot "${botchainName}". Please check your connection and try again.`;
                } else if (templateResult.error_type === 'missing_parameter') {
                    errorMessage = 'Please enter a bot name to continue.';
                }
                
                this.showStatus(errorMessage, 'error');
                document.getElementById('joinBtn').disabled = false;
                return;
            }
            
            // Template loaded successfully
            this.updateChatHeader(botchainName, 'Connecting...');
            this.updateStatusIndicator('connecting');
            this.updateConnectionStatus('connecting', `Loading ${botchainName}...`);
            
            // Get connection details
            const response = await this.fetchWithFallback(this.config.ENDPOINTS.CONNECTION_DETAILS, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    participant_name: this.participantName,
                    user_data: {
                        session_start: new Date().toISOString(),
                        botchain_name: botchainName,
                        org_name: orgName,
                        selected_voice: selectedVoice
                    }
                })
            });
            
            if (!response.ok) throw new Error('Failed to get connection details');
            const connectionDetails = await response.json();
            // Connection details received
            
            this.currentRoomName = connectionDetails.roomName;
            
            // Create room with proper configuration
            this.room = new LivekitClient.Room({
                adaptiveStream: true,
                dynacast: true,
                reconnectPolicy: {
                    nextRetryDelayInMs: (context) => {
                        // Reconnection attempt
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
            
            // Switch to chat interface
            this.showChatInterface();
            this.updateChatHeader(botchainName, 'Connected');
            this.updateStatusIndicator('connected');
            this.updateConnectionStatus('connected', 'Ready to listen');
            
            // Add welcome message
            this.addMessage(`Connected to ${botchainName}! I'm ready to help you.`, 'agent');
            
            this.initializeConversationLog();
            
            // Set up worker response timeout
            this.workerTimeout = setTimeout(() => {
                this.checkWorkerResponse();
            }, 20000); // 20 seconds timeout
            
        } catch (error) {
            console.error('Connection failed:', error);
            this.updateStatusIndicator('error');
            this.updateConnectionStatus('error', `Connection failed: ${error.message}`);
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
            // Room connected successfully
            this.isConnected = true;
            this.showStatus('Connected to Alive5 Dynamic Voice Support', 'connected');
            
            // Show chat interface instead of controls
            this.showChatInterface();
            
            const joinBtn = document.getElementById('joinBtn');
            if (joinBtn) {
                joinBtn.disabled = true;
            }
            
            this.createTranscriptContainer();
            
            // Update intent display to show "Detecting..." when connected
            this.updateIntentDisplay();
            
        });
        
        // Handle participant joining
        this.room.on(LivekitClient.RoomEvent.ParticipantConnected, (participant) => {
            // Participant joined
        });
        
        // Handle participant metadata changes (worker status signals)
        this.room.on(LivekitClient.RoomEvent.ParticipantMetadataChanged, (metadata, participant) => {
            if (participant.isAgent) {
                try {
                    // Check if metadata is valid JSON string
                    if (!metadata || metadata.trim() === '') {
                        return;
                    }
                    
                    const statusData = JSON.parse(metadata);
                    this.handleWorkerStatus(statusData);
                } catch (e) {
                    console.warn('Failed to parse worker status metadata:', e, 'Raw metadata:', metadata);
                }
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
            // Room disconnected
            this.isConnected = false;
            
            // Update intent display to show "Offline" when disconnected
            this.updateIntentDisplay();
            
            this.handleDisconnection();
        });
        
        // Handle audio tracks from agent
        this.room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
            // console.log('Track subscribed:', track.kind, 'from participant:', participant.identity);
            
            if (track.kind === 'audio') {
                let audioElement = document.getElementById(`audio-${participant.sid}`);
                if (!audioElement) {
                    audioElement = document.createElement('audio');
                    audioElement.id = `audio-${participant.sid}`;
                    audioElement.autoplay = true;
                    audioElement.controls = false;
                    document.body.appendChild(audioElement);
                }
                
                track.attach(audioElement);
                audioElement.volume = 1.0;
                
                try {
                    audioElement.play()
                        .then(() => {
                            console.log('Audio playback started');
                            // Clear worker timeout since agent audio is playing
                            this.clearWorkerTimeoutError();
                        })
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
        
        // ENHANCED: Register text stream handler for transcriptions with better real-time handling
        this.room.registerTextStreamHandler('lk.transcription', async (reader, participantInfo) => {
            try {
                const message = await reader.readAll();
                // console.log('📝 Transcription received:', { 
                //     message, 
                //     participantInfo: participantInfo.identity, 
                //     attributes: reader.info.attributes,
                //     myName: this.participantName 
                // });
                
                // More robust check for user vs agent transcription
                const isUserTranscript = reader.info.attributes && reader.info.attributes['lk.transcribed_track_id'];
                const isAgentIdentity = participantInfo.identity && (
                    participantInfo.identity.includes('agent') || 
                    participantInfo.identity.includes('Scott') ||
                    participantInfo.identity === 'Scott_AI_Agent'
                );
                
                // console.log('🔍 Transcript analysis:', { 
                //     isUserTranscript, 
                //     isAgentIdentity, 
                //     participantIdentity: participantInfo.identity,
                //     decision: isUserTranscript && !isAgentIdentity ? 'USER' : 'SKIP/AGENT'
                // });
                
                if (isUserTranscript && !isAgentIdentity) {
                    // This is definitely a user's speech transcription
                    console.log('👤 USER:', message);
                    
                    // Process all user transcripts immediately - no filtering
                    this.handleUserTranscription(message, participantInfo, reader.info.attributes);
                } else if (isAgentIdentity || (!isUserTranscript && participantInfo.identity !== this.participantName)) {
                    // This is likely agent speech - but we'll handle it via our custom agent transcript stream
                    // console.log('🤖 Scott:', message);
                    
                    // Clear worker timeout since agent is responding
                    this.clearWorkerTimeoutError();
                } else {
                    // Fallback - check content or other attributes
                    console.log('❓ Ambiguous transcript, treating as user:', message);
                    this.handleUserTranscription(message, participantInfo, reader.info.attributes);
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

        // ENHANCED: Register handler for agent transcript messages
        this.room.registerTextStreamHandler('lk.agent.transcript', async (reader, participantInfo) => {
            try {
                const data = await reader.readAll();
                console.log('Agent transcript data received:', data, 'from:', participantInfo.identity);
                
                try {
                    // Try to parse as JSON first (data format)
                    const parsed = JSON.parse(data);
                    if (parsed.type === 'agent_transcript') {
                        this.handleAgentTranscription(parsed.message, { identity: parsed.speaker });
                        return;
                    }
                } catch (e) {
                    // If not JSON, treat as plain text
                    this.handleAgentTranscription(data, { identity: 'Scott_AI_Agent' });
                }
            } catch (error) {
                console.error('Error processing agent transcript:', error);
            }
        });

        // Also register data handler for agent transcripts (fallback)
        this.room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant, kind, topic) => {
            if (topic === 'lk.agent.transcript') {
                try {
                    const data = JSON.parse(new TextDecoder().decode(payload));
                    if (data.type === 'agent_transcript') {
                        // console.log('🤖 Agent transcript via data received:', data);
                        this.handleAgentTranscription(data.message, { identity: data.speaker });
                    }
                } catch (error) {
                    console.error('Error processing agent transcript data:', error);
                }
            } else if (topic === 'lk.conversation.control') {
                try {
                    const data = JSON.parse(new TextDecoder().decode(payload));
                    if (data.type === 'conversation_end') {
                        console.log('🔚 Conversation end signal received:', data);
                        this.handleConversationEnd(data);
                        // Immediately show end UI and request disconnect
                        try {
                            this.showStatus('Conversation ended. Disconnecting...', 'info');
                            if (this.room) {
                                this.room.disconnect().catch(()=>{});
                            }
                        } catch(e) { /* ignore */ }
                    }
                } catch (error) {
                    console.error('Error processing conversation control data:', error);
                }
            } else if (topic === 'lk.intent.update') {
                try {
                    const data = JSON.parse(new TextDecoder().decode(payload));
                    if (data.type === 'intent_update' && data.intent) {
                        this.handleIntentUpdate(data.intent, data.source || 'Flow System');
                    }
                } catch (error) {
                    console.error('Error processing intent update data:', error);
                }
            } else if (topic) {
                console.log('📨 Other data received:', { topic, participant: participant?.identity, data: new TextDecoder().decode(payload) });
            }
        });
    }
    
    handleUserTranscription(transcriptText, participantInfo, attributes = {}) {
        //console.log('User transcription:', transcriptText, 'from:', participantInfo.identity);
        
        // Strictly treat only explicit finals as final; everything else is interim
        const isFinal = attributes['lk.transcript.final'] === 'true';
        
        // Update transcript display with proper interim/final handling
        this.updateTranscriptDisplay(transcriptText, isFinal);
        
        // Only add to conversation log and send for intent detection if it's final and complete
        if (isFinal && transcriptText.trim()) {
            const now = Date.now();
            const AGGREGATE_WINDOW_MS = 1200;

            // Clear any existing timeout
            if (this.transcriptTimeout) {
                clearTimeout(this.transcriptTimeout);
            }

            // Initialize last final timestamp storage
            if (typeof this.lastFinalAt !== 'number') {
                this.lastFinalAt = 0;
            }

            // If another final came in recently, aggregate instead of replacing
            if (now - this.lastFinalAt <= AGGREGATE_WINDOW_MS && this.pendingTranscript && this.pendingTranscript.length) {
                // Concatenate with a space if needed
                const joiner = this.pendingTranscript.endsWith(' ') ? '' : ' ';
                this.pendingTranscript = this.pendingTranscript + joiner + transcriptText;
            } else {
                // Start a new pending utterance
                this.pendingTranscript = transcriptText;
            }

            this.lastFinalAt = now;

            // Debounce processing to allow more finals to arrive and be aggregated
            this.transcriptTimeout = setTimeout(() => {
                this.processCompleteTranscript(this.pendingTranscript);
                this.pendingTranscript = '';
                this.lastFinalAt = 0;
            }, AGGREGATE_WINDOW_MS + 400);
        }
    }

    detectFarewell(message) {
        /**
         * Simple farewell detection on frontend (backup to worker detection)
         */
        const messageLower = message.toLowerCase().trim();
        const farewellKeywords = [
            'bye', 'goodbye', 'see you', 'that\'s all', 'thats all', 'i\'m done', 
            'im done', 'thank you bye', 'thanks bye', 'gotta go', 'have to go', 'okay, that\'s all', 'okay that\'s all', 'ok that\'s all', 'okay bye', 'ok bye', 'i think that\'s all', 'that\'s all goodbye', 'okay, thanks', 'okay, thanks bye', 'okay, thanks goodbye'
        ];
        
        return farewellKeywords.some(keyword => messageLower.includes(keyword));
    }
    
    handleAgentTranscription(transcriptText, participantInfo) {
        // console.log('Agent transcription:', transcriptText, 'from:', participantInfo.identity);
        
        // Ensure this is clearly marked as coming from the agent, not the user
        const agentIdentity = participantInfo.identity === 'Scott_AI_Agent' ? '🤖 Scott (AI)' : 
                             participantInfo.identity && participantInfo.identity.includes('Scott') ? '🤖 Scott (AI)' : 
                             '🤖 AI Assistant';
        
        // Add agent response to conversation log with clear agent identification
        this.addLogEntry({
            speaker: agentIdentity,
            message: transcriptText,
            type: 'agent'
        });
        
        // Update transcript display to show agent's speech
        const status = document.getElementById('transcript-status');
        if (status) {
            status.textContent = 'Scott responding...';
            status.className = 'agent-speaking';
            status.style.background = '#e3f2fd';
            status.style.color = '#1565c0';
            
            // Reset status after agent finishes speaking
            setTimeout(() => {
                if (status) {
                    status.textContent = 'Listening...';
                    status.className = 'listening';
                    status.style.background = '#e8f5e8';
                    status.style.color = '#2d5a2d';
                }
            }, 3000);
        }
    }

    handleConversationEnd(data) {
        console.log('🔚 Handling conversation end:', data);
        
        // Add system message about conversation ending
        this.addLogEntry({
            speaker: 'System',
            message: '👋 Conversation ended by user request. Disconnecting...',
            type: 'system'
        });
        
        // Show toast notification
        this.showToast('Conversation ended. Disconnecting...', 'info');
        
        // Update status
        this.showStatus('Conversation ended. Disconnecting...', 'info');
        
        // Disconnect after a short delay to allow farewell message to be heard
        setTimeout(() => {
            this.disconnect(true);
        }, 3000); // 3 second delay for polite farewell
    }
    
    async sendTranscriptForFlowProcessing(transcript) {
        try {
            // Get configuration values from form
            const botchainName = document.getElementById('botchainName')?.value?.trim() || null;
            const orgName = document.getElementById('orgName')?.value?.trim() || null;
            
            // Send conversation history to backend for flow processing
            const requestData = {
                room_name: this.currentRoomName,
                user_message: transcript,
                conversation_history: this.conversationHistory,
                botchain_name: botchainName,
                org_name: orgName
            };
            
            console.log('📤 Sending to backend:', {
                user_message: transcript,
                conversation_history_length: this.conversationHistory.length,
                conversation_history: this.conversationHistory
            });
            
            const response = await this.fetchWithFallback(this.config.ENDPOINTS.PROCESS_FLOW_MESSAGE, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Flow processing result:', result);
                
                // Clear worker timeout since backend is responding
                this.clearWorkerTimeoutError();
                
                // Handle flow result
                if (result.flow_result) {
                    this.handleFlowResult(result.flow_result);
                }
            }
        } catch (error) {
            console.warn('Failed to send transcript for flow processing:', error);
        }
    }
    
    handleFlowResult(flowResult) {
        const flowType = flowResult.type;
        const response = flowResult.response;
        
        console.log('Handling flow result:', flowType, response);
        
        // Add agent response to conversation history
        if (response && response.trim()) {
            this.conversationHistory.push({
                role: 'assistant',
                content: response,
                timestamp: new Date().toISOString()
            });
            
            // Keep only last 10 messages to avoid token limits
            if (this.conversationHistory.length > 10) {
                this.conversationHistory = this.conversationHistory.slice(-10);
            }
        }
        
        // Add flow information to conversation log
        this.addLogEntry({
            speaker: 'System',
            message: `Flow: ${flowType} - ${response}`,
            type: 'system'
        });
        
        // Handle different flow types
        switch (flowType) {
            case 'flow_started':
            const flowName = flowResult.flow_name || 'Unknown';
            this.handleIntentUpdate(flowName, 'Flow System');
                break;
                
            case 'agent_handoff':
                this.handleAgentHandoff(flowResult);
                break;
                
            case 'action_completed':
                this.handleActionCompleted(flowResult);
                break;
                
            case 'condition_evaluated':
                this.handleConditionEvaluated(flowResult);
                break;
                
            case 'greeting':
                this.handleGreeting(flowResult);
                break;
                
            case 'transfer_initiated':
                // Legacy escalation type - treat as agent handoff
                this.handleAgentHandoff(flowResult);
                break;
                
            default:
                // Handle other flow types
                if (flowType && flowType !== 'message') {
                    this.handleIntentUpdate(flowType, 'Bot System');
                }
                break;
        }
    }
    
    handleAgentHandoff(flowResult) {
        console.log('🤖 Agent handoff initiated:', flowResult);
        
        const escalationReason = flowResult.escalation_reason || 'unknown';
        const reasonText = this.getEscalationReasonText(escalationReason);
        
        // Show special UI for agent handoff
        this.addLogEntry({
            speaker: 'System',
            message: `👤 ${reasonText}`,
            type: 'system'
        });
        
        // Show toast notification
        this.showToast('Connecting to human agent...', 'info');
        
        // Update status
        this.showStatus('Agent handoff in progress...', 'info');
        
        // You could add additional logic here like:
        // - Notifying the backend about agent handoff
        // - Updating UI to show "Waiting for agent"
        // - Starting a timer for agent response
    }
    
    getEscalationReasonText(reason) {
        const reasonMap = {
            'user_requested_agent': 'User requested human agent - connecting...',
            'fallback_escalation': 'No intent found, escalating to human agent...',
            'faq_step_escalation': 'FAQ step escalation - connecting to human agent...',
            'unknown': 'Connecting to human agent...'
        };
        return reasonMap[reason] || reasonMap['unknown'];
    }
    
    handleActionCompleted(flowResult) {
        console.log('⚡ Action completed:', flowResult);
        
        const actionData = flowResult.action_data || {};
        
        // Show action completion in log
        this.addLogEntry({
            speaker: 'System',
            message: `⚡ Action completed: ${actionData.action_type || 'Unknown'}`,
            type: 'system'
        });
        
        // Show toast notification
        this.showToast('Action completed successfully!', 'success');
        
        // Handle specific action types
        if (actionData.url_opened) {
            this.addLogEntry({
                speaker: 'System',
                message: `🔗 URL opened: ${actionData.url_opened}`,
                type: 'system'
            });
        }
        
        if (actionData.email_sent) {
            this.addLogEntry({
                speaker: 'System',
                message: `📧 Email sent to: ${actionData.recipient}`,
                type: 'system'
            });
        }
    }
    
    handleConditionEvaluated(flowResult) {
        console.log('🔀 Condition evaluated:', flowResult);
        
        const conditionResult = flowResult.condition_result || {};
        
        // Show condition result in log
        this.addLogEntry({
            speaker: 'System',
            message: `🔀 Condition ${conditionResult.condition_met ? 'met' : 'not met'}: ${conditionResult.variable_name || 'Unknown'}`,
            type: 'system'
        });
        
        // Show toast notification
        const status = conditionResult.condition_met ? 'Condition met!' : 'Condition not met';
        this.showToast(status, conditionResult.condition_met ? 'success' : 'warning');
    }
    
    handleGreeting(flowResult) {
        console.log('👋 Greeting handled:', flowResult);
        
        // Show greeting in log
        this.addLogEntry({
            speaker: 'System',
            message: '👋 Greeting bot activated',
            type: 'system'
        });
        
        // Show toast notification
        this.showToast('Welcome! Greeting bot is active.', 'info');
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
            // Add final transcript with smooth transition
            const newFinalText = this.finalTranscript + (this.finalTranscript ? ' ' : '') + transcriptText;
            this.finalTranscript = newFinalText;
            this.interimTranscript = '';
            
            finalElement.textContent = this.finalTranscript;
            interimElement.textContent = '';
            
            // Add visual feedback for completed speech
            finalElement.classList.add('updated');
            setTimeout(() => finalElement.classList.remove('updated'), 1000);
        } else {
            // Show interim transcript with typing effect
            this.interimTranscript = transcriptText;
            interimElement.textContent = this.interimTranscript;
            interimElement.classList.add('typing');
        }
        
        // Update status with better feedback
        const status = document.getElementById('transcript-status');
        if (status) {
            if (!isFinal && transcriptText.trim()) {
                status.textContent = 'Speaking...';
                status.className = 'transcribing';
            } else if (isFinal) {
                status.textContent = 'Processing...';
                status.className = 'processing';
                // Reset to listening after a short delay
                setTimeout(() => {
                    if (status) {
                        status.textContent = 'Listening...';
                        status.className = 'listening';
                    }
                }, 1500);
            } else {
                status.textContent = 'Listening...';
                status.className = 'listening';
            }
        }
        
        // Auto-scroll transcript container
        const transcriptBody = document.getElementById('transcript-body');
        if (transcriptBody) {
            transcriptBody.scrollTop = transcriptBody.scrollHeight;
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
        container.style.cssText = `
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #e1e8ed;
            transition: all 0.3s ease;
        `;
        
        container.innerHTML = `
            <div class="transcript-header" style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.75rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #e1e8ed;
            ">
                <span style="font-weight: 600; color: #2c3e50;">🎤 Live Transcript</span>
                <span id="transcript-status" class="listening" style="
                    font-size: 0.85rem;
                    padding: 0.25rem 0.5rem;
                    border-radius: 12px;
                    background: #e8f5e8;
                    color: #2d5a2d;
                    font-weight: 500;
                ">Listening...</span>
            </div>
            <div id="transcript-body" class="transcript-body" style="
                max-height: 150px;
                overflow-y: auto;
                padding: 0.5rem;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
                font-family: 'Segoe UI', system-ui, sans-serif;
                line-height: 1.5;
            ">
                <div id="final-transcript" class="transcript-text final" style="
                    color: #2c3e50;
                    margin-bottom: 0.5rem;
                    transition: all 0.3s ease;
                "></div>
                <div id="interim-transcript" class="transcript-text interim" style="
                    color: #6c757d;
                    font-style: italic;
                    opacity: 0.8;
                    border-left: 3px solid #007bff;
                    padding-left: 0.5rem;
                    transition: all 0.3s ease;
                "></div>
            </div>
        `;
        
        // Since we don't have a controls element, we'll append to the chat container
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.appendChild(container);
        } else {
            document.querySelector('.container').appendChild(container);
        }
        
        // Add dynamic styles
        const style = document.createElement('style');
        style.textContent = `
            .transcript-text.updated {
                background-color: #e8f5e8 !important;
                transform: scale(1.02);
            }
            
            .transcript-text.typing {
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 0.8; }
                50% { opacity: 1; }
            }
            
            #transcript-status.transcribing {
                background: #fff3cd !important;
                color: #856404 !important;
            }
            
            #transcript-status.processing {
                background: #d1ecf1 !important;
                color: #0c5460 !important;
            }
            
            #transcript-status.listening {
                background: #e8f5e8 !important;
                color: #2d5a2d !important;
            }
        `;
        document.head.appendChild(style);
        
        // console.log('Enhanced transcript container created');
    }
    
    initializeConversationLog() {
        // Simple intent indicator in header - no complex analytics interface
        this.updateIntentDisplay();
    }
    
    updateIntentDisplay() {
        const intentDisplay = document.getElementById('intentDisplay');
        if (intentDisplay) {
            let intent;
            let className;
            
            if (!this.isConnected) {
                // When disconnected, show nothing or a neutral state
                intent = 'Offline';
                className = 'intent-offline';
            } else if (this.currentIntent) {
                // When connected and intent is detected
                intent = this.currentIntent;
                className = 'intent-detected';
            } else {
                // When connected but no intent detected yet
                intent = 'Detecting...';
                className = 'intent-detecting';
            }
            
            intentDisplay.textContent = intent;
            intentDisplay.className = className;
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
        
        // Add to chat interface if connected
        if (this.isConnected && (entry.type === 'user' || entry.type === 'agent')) {
            this.addMessage(entry.message, entry.type, entry.timestamp);
        }
        
        // Update intent display if this is an intent update
        if (entry.intent) {
            this.currentIntent = entry.intent;
            this.updateIntentDisplay();
        }
        
        // Simple console logging for debugging (no complex UI)
        console.log(`[${entry.type.toUpperCase()}] ${entry.speaker}: ${entry.message}`);
    }
    
    handleIntentUpdate(intent, source) {
        if (!intent) return;
        
        const oldIntent = this.currentIntent;
        this.currentIntent = intent;
        
        // Update the simple intent display in header
        this.updateIntentDisplay();
        
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
            await this.fetchWithFallback(this.config.ENDPOINTS.UPDATE_SESSION, {
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
                console.log('Disconnected from room:', this.currentRoomName);
                
                if (this.currentRoomName) {
                    this.fetchWithFallback(`${this.config.ENDPOINTS.DELETE_ROOM}/${this.currentRoomName}`, {
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
        this.currentRoomName = null;
        
        // Return to connection form
        this.showConnectionForm();
        this.updateStatusIndicator('offline');
        this.updateChatHeader('Voice Assistant', 'Ready to help');
        
        // Update intent display to show "Offline"
        this.updateIntentDisplay();
        
        // Clean up analytics interface elements
        const conversationLog = document.getElementById('conversationLog');
        if (conversationLog) {
            conversationLog.remove();
        }
        
        // Clean up transcript container
        const transcriptContainer = document.getElementById('transcript-container');
        if (transcriptContainer) {
            transcriptContainer.remove();
        }
        
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
        try {
            // Show reconnecting status
            this.showStatus('🔄 Reconnecting... Please wait', 'connecting');
            
            // Clean up existing connection
            if (this.room) {
                try {
                    await this.room.disconnect();
                } catch (e) {
                    console.warn('Error disconnecting during reconnect:', e);
                }
                this.room = null;
            }
            
            // Reset connection state
            this.isConnected = false;
            this.connectionAttempts = 0;
            
            // Clear any existing retry buttons
            const existingRetryBtn = document.querySelector('.retry-btn');
            if (existingRetryBtn) {
                existingRetryBtn.remove();
            }
            
            // Clear worker timeout
            if (this.workerTimeout) {
                clearTimeout(this.workerTimeout);
                this.workerTimeout = null;
            }
            
            // Hide controls
            this.hideControls();
            
            // Reconnect
            await this.joinRoom();
            
        } catch (error) {
            console.error('Reconnection failed:', error);
            this.showStatus(`Reconnection failed: ${error.message}`, 'error');
        }
    }
    
    showStatus(message, type) {
        // Prevent duplicate status messages within 100ms
        const statusKey = `${message}_${type}`;
        const now = Date.now();
        
        if (this._lastStatus && this._lastStatus.key === statusKey && (now - this._lastStatus.time) < 100) {
            return; // Skip duplicate status within 100ms
        }
        
        this._lastStatus = { key: statusKey, time: now };
        
        // Show toast notification for user feedback
        this.showToast(message, type);
        
        // Also log to console for debugging
        console.log(`Status (${type}): ${message}`);
        
        // Update the connection status in the chat interface if available
        const connectionStatus = document.getElementById('connectionStatus');
        if (connectionStatus) {
            connectionStatus.textContent = message;
            connectionStatus.className = `connection-status ${type}`;
        }
        
        // Also update the status text in the header if available
        const statusText = document.getElementById('statusText');
        if (statusText) {
            statusText.textContent = type === 'error' ? 'Error' : type === 'connecting' ? 'Connecting' : 'Offline';
        }
    }

    showToast(message, type = 'info', title = null) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            error: '❌',
            success: '✅',
            warning: '⚠️',
            info: 'ℹ️',
            connecting: '🔄'
        };

        const titles = {
            error: 'Error',
            success: 'Success',
            warning: 'Warning',
            info: 'Info',
            connecting: 'Connecting'
        };

        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-content">
                <div class="toast-title">${title || titles[type] || 'Notification'}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;

        container.appendChild(toast);

        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 10);

        // Auto remove after different times based on type
        const autoRemoveTime = type === 'error' ? 8000 : 
                               type === 'info' || type === 'success' ? 2000 : 5000;
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }
        }, autoRemoveTime);
    }

    // New chatbot interface methods
    updateChatHeader(botName, status) {
        const botNameElement = document.getElementById('botName');
        const botStatusElement = document.getElementById('botStatus');
        
        if (botNameElement) {
            botNameElement.textContent = botName || 'Voice Assistant';
        }
        
        if (botStatusElement) {
            botStatusElement.textContent = status || 'Ready to help';
        }
    }

    updateStatusIndicator(status) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        statusDot.className = 'status-dot';
        switch(status) {
            case 'connected':
                statusDot.classList.add('connected');
                statusText.textContent = 'Online';
                break;
            case 'connecting':
                statusDot.classList.add('connecting');
                statusText.textContent = 'Connecting...';
                break;
            case 'error':
                statusDot.classList.add('error');
                statusText.textContent = 'Error';
                break;
            default:
                statusText.textContent = 'Offline';
        }
    }

    showChatInterface() {
        document.getElementById('connectionForm').style.display = 'none';
        document.getElementById('chatMessages').style.display = 'flex';
        document.getElementById('chatInputArea').style.display = 'flex';
        
        // Show transcript container when in chat interface
        const transcriptContainer = document.getElementById('transcript-container');
        if (transcriptContainer) {
            transcriptContainer.style.display = 'block';
        }
        
        // Sync voice selectors
        const voiceChangeSelect = document.getElementById('voiceChangeSelect');
        if (voiceChangeSelect) {
            voiceChangeSelect.value = this.selectedVoice;
        }
    }

    showConnectionForm() {
        document.getElementById('connectionForm').style.display = 'flex';
        document.getElementById('chatMessages').style.display = 'none';
        document.getElementById('chatInputArea').style.display = 'none';
        
        // Hide transcript container when returning to connection form
        const transcriptContainer = document.getElementById('transcript-container');
        if (transcriptContainer) {
            transcriptContainer.style.display = 'none';
        }
    }

    addMessage(content, sender, timestamp = null) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            console.warn('Chat messages container not found');
            return;
        }
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : '🤖';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = content;
        
        if (timestamp) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date(timestamp).toLocaleTimeString();
            bubble.appendChild(timeDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message agent';
        typingDiv.id = 'typingIndicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🤖';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        
        const dots = document.createElement('div');
        dots.className = 'typing-dots';
        dots.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        
        indicator.appendChild(dots);
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(indicator);
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    updateConnectionStatus(status, message) {
        const statusEl = document.getElementById('connectionStatus');
        if (statusEl) {
            statusEl.textContent = message || status;
            statusEl.className = `connection-status ${status}`;
        }
    }

    showVoiceVisualizer(show = true) {
        const visualizer = document.getElementById('voiceVisualizer');
        if (visualizer) {
            visualizer.style.display = show ? 'flex' : 'none';
        }
    }

    async changeVoice(voiceId) {
        if (!this.isConnected || !this.currentRoomName) {
            console.warn('Cannot change voice: not connected');
            return;
        }

        try {
            console.log(`🎤 VOICE_CHANGE: Attempting to change voice to ${voiceId} for room ${this.currentRoomName}`);
            console.log(`🎤 VOICE_CHANGE: Voice name: ${this.getVoiceName(voiceId)}`);
            this.updateConnectionStatus('connecting', 'Changing voice...');
            
            const requestBody = {
                room_name: this.currentRoomName,
                voice_id: voiceId
            };
            console.log(`🎤 VOICE_CHANGE: Request body:`, requestBody);
            
            const response = await this.fetchWithFallback('/api/change_voice', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            console.log(`🎤 VOICE_CHANGE: Response status: ${response.status}`);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error(`🎤 VOICE_CHANGE: Response error: ${errorText}`);
                throw new Error(`Failed to change voice: ${response.status} ${errorText}`);
            }

            const result = await response.json();
            console.log(`🎤 VOICE_CHANGE: Response result:`, result);
            
            if (result.status === 'success') {
                this.selectedVoice = voiceId;
                const voiceName = result.voice_name || this.getVoiceName(voiceId);
                this.addMessage(`Voice changed to ${voiceName}`, 'system');
                this.updateConnectionStatus('connected', 'Voice changed successfully');
                console.log(`🎤 VOICE_CHANGE: Successfully changed voice to ${voiceId} (${voiceName})`);
            } else {
                throw new Error(result.message || 'Failed to change voice');
            }
        } catch (error) {
            console.error('Error changing voice:', error);
            this.addMessage(`Failed to change voice: ${error.message}`, 'system');
            this.updateConnectionStatus('error', 'Voice change failed');
        }
    }

    getVoiceName(voiceId) {
        const voiceNames = {
            'a167e0f3-df7e-4d52-a9c3-f949145efdab': 'Customer Support Man (Default)',
            '3f4ade23-6eb4-4279-ab05-6a144947c4d5': 'German Conversational Woman',
            '79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e': 'Nonfiction Man',
            'e00d0e4c-a5c8-443f-a8a3-473eb9a62355': 'Friendly Sidekick',
            'a249eaff-1e96-4d2c-b23b-12efa4f66f41': 'French Conversational Lady',
            '8832a0b5-47b2-4751-bb22-6a8e2149303d': 'French Narrator Lady',
            '119e03e4-0705-43c9-b3ac-a658ce2b6639': 'German Reporter Woman',
            '3b554273-4299-48b9-9aaf-eefd438e3941': 'Indian Lady',
            '71a7ad14-091c-4e8e-a314-022ece01c121': 'British Reading Lady',
            '4d2fd738-3b3d-4368-957a-bb4805275bd9': 'British Narration Lady',
            '44863732-e415-4084-8ba1-deabe34ce3d2': 'Japanese Children Book',
            '2b568345-1d48-4047-b25f-7baccf842eb0': 'Japanese Woman Conversational',
            'e8a863c6-22c7-4671-86ca-91cacffc038d': 'Japanese Male Conversational',
            '15a9cd88-84b0-4a8b-95f2-5d583b54c72e': 'Reading Lady',
            'd46abd1d-2d02-43e8-819f-51fb652c1c61': 'Newsman',
            '2ee87190-8f84-4925-97da-e52547f9462c': 'Child',
            'cd17ff2d-5ea4-4695-be8f-42193949b946': 'Meditation Lady',
            '5345cf08-6f37-424d-a5d9-8ae1101b9377': 'Maria',
            '41534e16-2966-4c6b-9670-111411def906': '1920\'s Radioman',
            'bf991597-6c13-47e4-8411-91ec2de5c466': 'Newslady',
            '00a77add-48d5-4ef6-8157-71e5437b282d': 'Calm Lady',
            '156fb8d2-335b-4950-9cb3-a2d33befec77': 'Helpful Woman',
            '5c5ad5e7-1020-476b-8b91-fdcbe9cc313c': 'Mexican Woman',
            'b7d50908-b17c-442d-ad8d-810c63997ed9': 'California Girl',
            '663afeec-d082-4ab5-827e-2e41bf73a25b': 'Korean Narrator Woman',
            '779673f3-895f-4935-b6b5-b031dc78b319': 'Russian Calm Lady',
            '2b3bb17d-26b9-421f-b8ca-1dd92332279f': 'Russian Narrator Man 1',
            'da05e96d-ca10-4220-9042-d8acef654fa9': 'Russian Narrator Man 2',
            '642014de-c0e3-4133-adc0-36b5309c23e6': 'Russian Narrator Woman',
            '95d51f79-c397-46f9-b49a-23763d3eaa2d': 'Hinglish Speaking Lady',
            '0e21713a-5e9a-428a-bed4-90d410b87f13': 'Italian Narrator Woman',
            '575a5d29-1fdc-4d4e-9afa-5a9a71759864': 'Polish Narrator Woman',
            'e90c6678-f0d3-4767-9883-5d0ecf5894a8': 'Chinese Female Conversational',
            '36b42fcb-60c5-4bec-b077-cb1a00a92ec6': 'Pilot over Intercom',
            'eda5bbff-1ff1-4886-8ef1-4e69a77640a0': 'Chinese Commercial Man',
            '5c3c89e5-535f-43ef-b14d-f8ffe148c1f0': 'French Narrator Man',
            'a67e0421-22e0-4d5b-b586-bd4a64aee41d': 'Spanish Narrator Man',
            'f146dcec-e481-45be-8ad2-96e1e40e7f32': 'Reading Man',
            '34575e71-908f-4ab6-ab54-b08c95d6597d': 'New York Man',
            'ab7c61f5-3daa-47dd-a23b-4ac0aac5f5c3': 'Friendly French Man',
            'a0e99841-438c-4a64-b679-ae501e7d6091': 'Barbershop Man',
            '638efaaa-4d0c-442e-b701-3fae16aad012': 'Indian Man',
            '41f3c367-e0a8-4a85-89e0-c27bae9c9b6d': 'Australian Customer Support Man',
            '421b3369-f63f-4b03-8980-37a44df1d4e8': 'Friendly Australian Man',
            'b043dea0-a007-4bbe-a708-769dc0d0c569': 'Wise Man',
            '6926713a-4b0c-4b0c-4b0c-4b0c4b0c4b0c': 'Friendly Reading Man',
            '3f6e78a8-5283-42aa-b5e7-af82e8bb310c': 'German Reporter Man',
            '63ff761f-c1e8-414b-b969-d1833d1c870c': 'Confident British Man',
            '98a34ef2-2140-4c28-9c71-663dc4dd7022': 'Southern Man',
            '95856005-0332-41b0-935f-352e296aa0df': 'Classy British Man',
            'ee7ea9f8-c0c1-498c-9279-764d6b56d189': 'Polite Man',
            '15d0c2e2-8d29-44c3-be23-d585d5f154a1': 'Mexican Man',
            '57dba6ff-fe3b-479d-836e-06f5a61cb5de': 'Korean Narrator Man',
            '5a31e4fb-f823-4359-aa91-82c0ae9a991c': 'Turkish Narrator Man',
            '39f753ef-b0eb-41cd-aa53-2f3c284f948f': 'Turkish Calm Man',
            'ac7ee4fa-25db-420d-bfff-f590d740aeb2': 'Hindi Calm Man',
            '7f423809-0011-4658-ba48-a411f5e516ba': 'Hindi Narrator Man',
            '4ef93bb3-682a-46e6-b881-8e157b6b4388': 'Polish Narrator Man',
            '82a7fc13-2927-4e42-9b8a-bb1f9e506521': 'Polish Young Man',
            '40104aff-a015-4da1-9912-af950fbec99e': 'Alabama Male',
            '13524ffb-a918-499a-ae97-c98c7c4408c4': 'Australian Male',
            '1001d611-b1a8-46bd-a5ca-551b23505334': 'Anime Girl',
            '97e7d7a9-dfaa-4758-a936-f5f844ac34cc': 'Japanese Man Book',
            'e3827ec5-697a-4b7c-9704-1a23041bbc51': 'Sweet Lady',
            'c2ac25f9-ecc4-4f56-9095-651354df60c0': 'Commercial Lady',
            '573e3144-a684-4e72-ac2b-9b2063a50b53': 'Teacher Lady',
            '8f091740-3df1-4795-8bd9-dc62d88e5131': 'Princess',
            '7360f116-6306-4e9a-b487-1235f35a0f21': 'Commercial Man',
            '03496517-369a-4db1-8236-3d3ae459ddf7': 'ASMR Lady',
            '248be419-c632-4f23-adf1-5324ed7dbf1d': 'Professional Woman',
            'bd9120b6-7761-47a6-a446-77ca49132781': 'Tutorial Man',
            'a8a1eb38-5f15-4c1d-8722-7ac0f329727d': 'Calm French Woman',
            '34bde396-9fde-4ebf-ad03-e3a1d1155205': 'New York Woman',
            '846d6cb0-2301-48b6-9683-48f5618ea2f6': 'Spanish-speaking Lady',
            '11af83e2-23eb-452f-956e-7fee218ccb5c': 'Midwestern Woman',
            'ed81fd13-2016-4a49-8fe3-c0d2761695fc': 'Sportsman',
            '996a8b96-4804-46f0-8e05-3fd4ef1a87cd': 'Storyteller Lady',
            '34dbb662-8e98-413c-a1ef-1a3407675fe7': 'Spanish-speaking Man',
            'fb26447f-308b-471e-8b00-8e9f04284eb5': 'Doctor Mischief',
            '2695b6b5-5543-4be1-96d9-3967fb5e7fec': 'Spanish-speaking Reporter Man',
            'db832ebd-3cb6-42e7-9d47-912b425adbaa': 'Young Spanish-speaking Woman',
            '50d6beb4-80ea-4802-8387-6c948fe84208': 'The Merchant',
            '0418348a-0ca2-4e90-9986-800fb8b3bbc0': 'Stern French Man',
            'e13cae5c-ec59-4f71-b0a6-266df3c9bb8e': 'Madame Mischief',
            'db229dfe-f5de-4be4-91fd-7b077c158578': 'German Storyteller Man',
            '5c42302c-194b-4d0c-ba1a-8cb485c84ab9': 'Female Nurse',
            '384b625b-da5d-49e8-a76d-a2855d4f31eb': 'German Conversation Man',
            '6a16c1f4-462b-44de-998d-ccdaa4125a0a': 'Friendly Brazilian Man',
            'b9de4a89-2257-424b-94c2-db18ba68c81a': 'German Woman',
            'f9836c6e-a0bd-460e-9d3c-f7299fa60f94': 'Southern Woman',
            'a01c369f-6d2d-4185-bc20-b32c225eab70': 'British Customer Support Lady',
            'd4d4b115-57a0-48ea-9a1a-9898966c2966': 'Chinese Woman Narrator'
        };
        return voiceNames[voiceId] || 'Unknown Voice';
    }
    
    showControls() {
        // Since we don't have a controls element, we'll show the chat interface
        this.showChatInterface();
    }
    
    checkWorkerResponse() {
        // Check if we've received any agent responses
        const hasAgentResponse = this.conversationLog.some(log => 
            log.type === 'agent' || 
            log.sender === 'Scott' ||
            log.speaker?.includes('Scott') ||
            log.speaker?.includes('agent')
        );
        
        if (!hasAgentResponse) {
            this.showStatus(`
                ⚠️ Worker Connection Issue<br>
                The voice agent is not responding. This may be due to:<br>
                • Backend server not running<br>
                • Worker initialization timeout<br>
                • Network connectivity issues<br><br>
                <strong>Please try:</strong><br>
                1. Refresh the page and try again<br>
                2. Check if the backend is running<br>
                3. Contact support if the issue persists
            `, 'error');
            
            // Add a retry button
            this.addRetryButton();
        } else {
            // Agent has responded, clear any existing error status
            console.log('✅ Agent response detected, clearing worker timeout error');
            this.clearWorkerTimeoutError();
        }
    }
    
    clearWorkerTimeoutError() {
        // Clear the worker timeout
        if (this.workerTimeout) {
            clearTimeout(this.workerTimeout);
            this.workerTimeout = null;
        }
        
        // Remove any existing retry buttons
        const existingRetryBtn = document.querySelector('.retry-btn');
        if (existingRetryBtn) {
            existingRetryBtn.remove();
        }
        
        // Clear error status if it's showing worker connection issue
        // Since we removed the status element, we'll just log this
        // console.log('Clearing worker connection issue status');
    }
    
    addRetryButton() {
        // Remove existing retry button if any
        const existingRetryBtn = document.querySelector('.retry-btn');
        if (existingRetryBtn) {
            existingRetryBtn.remove();
        }
        
        // Add new retry button
        const retryBtn = document.createElement('button');
        retryBtn.textContent = '🔄 Retry Connection';
        retryBtn.className = 'retry-btn';
        retryBtn.onclick = async () => {
            // Disable button and show loading
            retryBtn.disabled = true;
            retryBtn.textContent = '🔄 Retrying...';
            retryBtn.style.opacity = '0.6';
            
            try {
                await this.reconnect();
            } catch (error) {
                // Re-enable button if reconnection fails
                retryBtn.disabled = false;
                retryBtn.textContent = '🔄 Retry Connection';
                retryBtn.style.opacity = '1';
            }
        };
        
        // Since we removed the status element, we'll append the retry button to the form container
        const formContainer = document.querySelector('.form-container');
        if (formContainer) {
            formContainer.appendChild(retryBtn);
        }
    }
    
    handleWorkerStatus(statusData) {
        const { worker_status, message, timestamp } = statusData;
        console.log('Worker status update:', statusData);
        
        switch (worker_status) {
            case 'ready':
                this.showStatus(`
                    ✅ Worker Ready<br>
                    ${message}<br>
                    <small>Connected at: ${new Date(timestamp).toLocaleTimeString()}</small>
                `, 'connected');
                break;
                
            case 'backend_down':
                this.showStatus(`
                    ⚠️ Backend Server Down<br>
                    ${message}<br>
                    <small>Worker is trying to reconnect automatically...</small>
                `, 'warning');
                
                // Add retry button for manual retry
                this.addRetryButton();
                break;
                
            case 'retrying':
                this.showStatus(`
                    🔄 Reconnecting...<br>
                    ${message}<br>
                    <small>Attempting to restore connection...</small>
                `, 'connecting');
                break;
                
            case 'reconnected':
                this.showStatus(`
                    ✅ Connection Restored<br>
                    ${message}<br>
                    <small>Backend is back online!</small>
                `, 'connected');
                break;
                
            case 'failed':
                this.showStatus(`
                    ❌ Connection Failed<br>
                    ${message}<br>
                    <small>Please refresh the page and try again</small>
                `, 'error');
                
                // Add retry button for manual retry
                this.addRetryButton();
                break;
                
            default:
                console.log('Unknown worker status:', worker_status);
        }
    }
    
    hideControls() {
        // Since we don't have a controls element, we'll hide the chat interface
        document.getElementById('connectionForm').style.display = 'block';
        document.getElementById('chatMessages').style.display = 'none';
        document.getElementById('chatInputArea').style.display = 'none';
        
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
            transcriptContainer.style.display = 'none';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DynamicVoiceAgent();
});