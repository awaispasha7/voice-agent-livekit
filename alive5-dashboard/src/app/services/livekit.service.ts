/**
 * HITL Voice Handoff - LiveKit Service
 * Handles LiveKit room connection and audio streaming
 */

import { Injectable } from '@angular/core';
import { 
  Room, 
  RoomEvent, 
  LocalAudioTrack, 
  RemoteAudioTrack, 
  RemoteParticipant,
  createLocalAudioTrack 
} from 'livekit-client';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class LiveKitService {
  private room: Room | null = null;
  private localAudioTrack: LocalAudioTrack | null = null;
  
  private connectedSubject = new BehaviorSubject<boolean>(false);
  public connected$: Observable<boolean> = this.connectedSubject.asObservable();
  
  private mutedSubject = new BehaviorSubject<boolean>(false);
  public muted$: Observable<boolean> = this.mutedSubject.asObservable();

  constructor() {}

  /**
   * Join a LiveKit room as a human agent
   */
  async joinRoom(token: string, roomName: string, livekitUrl: string): Promise<void> {
    try {
      console.log(`[LiveKit] Joining room: ${roomName}`);
      
      // Create room instance
      this.room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      // Set up event listeners
      this.setupEventListeners();

      // Connect to room
      await this.room.connect(livekitUrl, token);
      console.log('[LiveKit] Connected to room');

      // Publish microphone audio
      await this.publishMicrophone();
      
      this.connectedSubject.next(true);
    } catch (error) {
      console.error('[LiveKit] Failed to join room:', error);
      throw error;
    }
  }

  /**
   * Publish microphone audio to the room
   */
  private async publishMicrophone(): Promise<void> {
    try {
      if (!this.room) {
        throw new Error('Room not connected');
      }

      // Create local audio track from microphone
      this.localAudioTrack = await createLocalAudioTrack({
        autoGainControl: true,
        echoCancellation: true,
        noiseSuppression: true,
      });

      // Publish to room
      await this.room.localParticipant.publishTrack(this.localAudioTrack);
      console.log('[LiveKit] Microphone published');
    } catch (error) {
      console.error('[LiveKit] Failed to publish microphone:', error);
      throw error;
    }
  }

  /**
   * Set up room event listeners
   */
  private setupEventListeners(): void {
    if (!this.room) return;

    // Handle remote participant audio
    this.room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (track.kind === 'audio') {
        console.log(`[LiveKit] Subscribed to audio from ${participant.identity}`);
        // Audio will automatically play through browser
        const audioElement = track.attach();
        document.body.appendChild(audioElement);
      }
    });

    // Handle participant disconnections
    this.room.on(RoomEvent.ParticipantDisconnected, (participant: RemoteParticipant) => {
      console.log(`[LiveKit] Participant disconnected: ${participant.identity}`);
    });

    // Handle disconnection
    this.room.on(RoomEvent.Disconnected, () => {
      console.log('[LiveKit] Disconnected from room');
      this.connectedSubject.next(false);
    });
  }

  /**
   * Toggle microphone mute
   */
  async toggleMute(): Promise<void> {
    if (!this.localAudioTrack) {
      console.warn('[LiveKit] No local audio track to mute');
      return;
    }

    const currentlyMuted = this.localAudioTrack.isMuted;
    
    if (currentlyMuted) {
      await this.localAudioTrack.unmute();
    } else {
      await this.localAudioTrack.mute();
    }
    
    const newMutedState = !currentlyMuted;
    this.mutedSubject.next(newMutedState);
    console.log(`[LiveKit] Microphone ${newMutedState ? 'muted' : 'unmuted'}`);
  }

  /**
   * Leave the room and cleanup
   */
  async leaveRoom(): Promise<void> {
    try {
      console.log('[LiveKit] Leaving room');
      
      if (this.localAudioTrack) {
        this.localAudioTrack.stop();
        this.localAudioTrack = null;
      }

      if (this.room) {
        await this.room.disconnect();
        this.room = null;
      }

      this.connectedSubject.next(false);
      this.mutedSubject.next(false);
    } catch (error) {
      console.error('[LiveKit] Error leaving room:', error);
      throw error;
    }
  }

  /**
   * Get current room state
   */
  isConnected(): boolean {
    return this.room !== null && this.room.state === 'connected';
  }

  /**
   * Get current mute state
   */
  isMuted(): boolean {
    return this.localAudioTrack?.isMuted ?? false;
  }
}

