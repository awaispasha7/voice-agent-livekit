class VoiceAgent {
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
        
        this.bindEvents();
        this.updateRoomDisplay();
    }
    
    // Generate unique session ID for each user
    generateSessionId() {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substring(2, 8);
        return `session_${timestamp}_${random}`;
    }
    
    updateRoomDisplay() {
        // Hide the room name input since we're auto-generating it
        const roomInput = document.getElementById('roomName');
        if (roomInput) {
            roomInput.style.display = 'none';
            // Also hide the label if it exists
            const label = document.querySelector('label[for="roomName"]');
            if (label) label.style.display = 'none';
        }
        
        // Show session info
        const statusEl = document.getElementById('status');
        if (statusEl) {
            statusEl.innerHTML = `Session ID: <code>${this.sessionId}</code><br>Ready to connect to Alive5 Support...`;
            statusEl.className = 'status info';
            statusEl.style.display = 'block';
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
        
        // Add reconnect functionality
        document.addEventListener('keydown', (e) => {
            if (e.key === 'r' && e.ctrlKey && !this.isConnected) {
                e.preventDefault();
                this.reconnect();
            }
        });
        
        // Handle page unload to cleanup
        window.addEventListener('beforeunload', () => {
            if (this.isConnected) {
                this.disconnect(false); // Silent disconnect on page unload
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
            this.showStatus('Connecting to Alive5 Support...', 'connecting');
            document.getElementById('joinBtn').disabled = true;
            
            // Get connection details - let backend generate unique room
            const response = await fetch('http://localhost:8000/api/connection_details', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    participant_name: this.participantName
                    // No room_name - let backend generate unique one
                })
            });
            
            if (!response.ok) throw new Error('Failed to get connection details');
            const connectionDetails = await response.json();
            console.log('Connection details:', connectionDetails);
            
            this.currentRoomName = connectionDetails.roomName;
            
            // Create room with optimized settings for voice chat
            this.room = new LivekitClient.Room({
                adaptiveStream: true,
                dynacast: true,
                reconnectPolicy: {
                    nextRetryDelayInMs: (context) => {
                        console.log('Reconnection attempt:', context.retryCount);
                        return Math.min(1000 * Math.pow(2, context.retryCount), 30000);
                    },
                    maxRetries: 3,
                },
            });
            
            this.setupRoomEvents();
            
            // Connect to room
            await this.room.connect(connectionDetails.serverUrl, connectionDetails.participantToken);
            await this.enableMicrophone();
            
            this.isConnected = true;
            this.connectionAttempts = 0; // Reset on successful connection
            this.showStatus(`🔗 Connected to Alive5 Support!<br>Room: ${this.currentRoomName}<br>⏳ Waiting for Ileana to join... (This may take 10-15 seconds)`, 'connected');
            this.showControls();
            
        } catch (error) {
            console.error('Connection failed:', error);
            this.showStatus(`Connection failed: ${error.message}`, 'error');
            document.getElementById('joinBtn').disabled = false;
            
            // Cleanup on failed connection
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
        // When any track is subscribed (agent speaking)
        this.room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
            console.log('Track subscribed:', track.kind, 'from', participant.identity);
            if (participant.sid === this.room.localParticipant.sid) return;
            
            if (track.kind === LivekitClient.Track.Kind.Audio) {
                console.log('Setting up agent audio...');
                const audioElement = track.attach();
                audioElement.style.display = 'none';
                audioElement.volume = 1.0; // Ensure full volume
                document.body.appendChild(audioElement);
                
                // Ensure audio plays with better error handling
                const playAudio = async () => {
                    try {
                        await audioElement.play();
                        console.log('Agent audio is playing successfully');
                    } catch (e) {
                        console.error('Audio play failed:', e);
                        if (e.name === 'NotAllowedError') {
                            this.showStatus('⚠️ Please click anywhere to enable audio playback', 'warning');
                            // Try to play again after user interaction
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
                
                console.log('Agent audio track attached');
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.Connected, () => {
            console.log('✅ Connected to room:', this.currentRoomName);
            console.log('Local participant:', this.room.localParticipant.identity);
        });
        
        this.room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
            console.log('❌ Disconnected from room. Reason:', reason);
            this.handleDisconnection();
        });
        
        this.room.on(LivekitClient.RoomEvent.ParticipantConnected, (participant) => {
            console.log('Participant connected:', participant.identity);
            if (participant.identity.toLowerCase().includes('agent') || 
                participant.identity.toLowerCase().includes('ileana')) {
                this.showStatus('🎤 Ileana has joined! She can hear you now - start talking!', 'connected');
            }
        });
        
        this.room.on(LivekitClient.RoomEvent.ParticipantDisconnected, (participant) => {
            console.log('❌ Participant disconnected:', participant.identity);
            if (participant.identity.toLowerCase().includes('agent') || 
                participant.identity.toLowerCase().includes('ileana')) {
                this.showStatus('⚠️ Ileana has left the conversation. The session may have ended.', 'warning');
            }
        });
        
        // Handle connection quality
        this.room.on(LivekitClient.RoomEvent.ConnectionQualityChanged, (quality, participant) => {
            if (participant === this.room.localParticipant) {
                console.log('Connection quality:', quality);
                if (quality === 'poor') {
                    this.showStatus('Connection quality is poor. Audio may be affected.', 'warning');
                }
            }
        });
        
        // Handle reconnection
        this.room.on(LivekitClient.RoomEvent.Reconnecting, () => {
            this.showStatus('Connection lost. Reconnecting...', 'connecting');
        });
        
        this.room.on(LivekitClient.RoomEvent.Reconnected, () => {
            this.showStatus('Reconnected successfully!', 'connected');
        });
    }
    
    handleDisconnection() {
        this.isConnected = false;
        this.hideControls();
        this.currentRoomName = null;
        
        // Cleanup audio elements
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(el => el.remove());
        
        this.showStatus('Disconnected from Alive5 Support. Click "Join Voice Chat" to start a new session.', 'error');
        document.getElementById('joinBtn').disabled = false;
        document.getElementById('joinBtn').textContent = 'Join Voice Chat';
        
        // Generate new session ID for next connection
        this.sessionId = this.generateSessionId();
        this.updateRoomDisplay();
    }
    
    async reconnect() {
        if (!this.isConnected) {
            this.connectionAttempts = 0; // Reset attempts for manual reconnect
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
    }
    
    async disconnect(showMessage = true) {
        if (!this.room) return;
        
        try {
            // Clean disconnect
            if (this.isConnected) {
                await this.room.disconnect();
                console.log('Manually disconnected from room:', this.currentRoomName);
                
                // Notify backend about room cleanup
                if (this.currentRoomName) {
                    fetch(`http://localhost:8000/api/rooms/${this.currentRoomName}`, {
                        method: 'DELETE'
                    }).catch(e => console.warn('Room cleanup notification failed:', e));
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
        
        // Reset mute button
        const muteBtn = document.getElementById('muteBtn');
        muteBtn.textContent = '🔇 Mute';
        muteBtn.classList.remove('muted');
        this.isMuted = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VoiceAgent();
});