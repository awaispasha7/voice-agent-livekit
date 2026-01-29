/**
 * HITL Voice Handoff - Call State Service
 * Manages active call state and coordinates between LiveKit and Alive5 Socket
 */

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { CallSession } from '../models/call.model';
import { LiveKitService } from './livekit.service';
import { Alive5SocketService } from './alive5-socket.service';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class CallStateService {
  private activeCallSubject = new BehaviorSubject<CallSession | null>(null);
  public activeCall$: Observable<CallSession | null> = this.activeCallSubject.asObservable();
  
  private callDurationSubject = new BehaviorSubject<number>(0);
  public callDuration$: Observable<number> = this.callDurationSubject.asObservable();
  
  private durationInterval: any = null;
  
  private backendUrl: string = environment.backendUrl;

  constructor(
    private http: HttpClient,
    private liveKitService: LiveKitService,
    private alive5SocketService: Alive5SocketService
  ) {}

  /**
   * Accept an incoming call
   */
  async acceptCall(call: any, agentId: string, agentName: string): Promise<void> {
    try {
      console.log('[CallState] Accepting call:', call.room_name);
      
      // Step 1: Request takeover
      await this.http.post(`${this.backendUrl}/api/human-agent/request-takeover`, {
        room_name: call.room_name,
        agent_id: agentId,
        agent_name: agentName
      }).toPromise();
      
      // Step 2: Generate LiveKit token
      const tokenResponse: any = await this.http.post(`${this.backendUrl}/api/human-agent/generate-token`, {
        room_name: call.room_name,
        agent_id: agentId
      }).toPromise();
      
      // Step 3: Join LiveKit room
      await this.liveKitService.joinRoom(
        tokenResponse.token,
        tokenResponse.room_name,
        tokenResponse.livekit_url
      );
      
      // Step 4: Set active call
      const session: CallSession = {
        room_name: call.room_name,
        thread_id: call.thread_id,
        crm_id: call.crm_id,
        channel_id: tokenResponse.session_data.channel_id,
        caller_phone: call.caller_phone,
        queue: call.queue,
        agent_id: agentId,
        agent_name: agentName,
        started_at: new Date()
      };
      
      this.activeCallSubject.next(session);
      
      // Step 5: Start duration timer
      this.startDurationTimer();
      
      // Step 6: Remove from incoming queue
      this.alive5SocketService.removeIncomingCall(call.room_name);
      
      console.log('[CallState] Call accepted and connected');
    } catch (error) {
      console.error('[CallState] Failed to accept call:', error);
      throw error;
    }
  }

  /**
   * Reject/decline an incoming call (no LiveKit join)
   */
  async rejectCall(call: any, agentId: string, agentName: string, reason: string = 'rejected'): Promise<void> {
    try {
      console.log('[CallState] Rejecting call:', call.room_name);

      await this.http.post(`${this.backendUrl}/api/human-agent/reject-takeover`, {
        room_name: call.room_name,
        agent_id: agentId,
        agent_name: agentName,
        reason
      }).toPromise();

      // Remove from incoming queue
      this.alive5SocketService.removeIncomingCall(call.room_name);
    } catch (error) {
      console.error('[CallState] Failed to reject call:', error);
      throw error;
    }
  }

  /**
   * End the active call
   */
  async endCall(resumeAI: boolean = false): Promise<void> {
    try {
      const activeCall = this.activeCallSubject.value;
      if (!activeCall) {
        console.warn('[CallState] No active call to end');
        return;
      }
      
      console.log('[CallState] Ending call:', activeCall.room_name);
      
      // Step 1: Notify backend
      await this.http.post(`${this.backendUrl}/api/human-agent/end-handoff`, {
        room_name: activeCall.room_name,
        agent_id: activeCall.agent_id,
        resume_ai: resumeAI
      }).toPromise();
      
      // Step 2: Leave LiveKit room
      await this.liveKitService.leaveRoom();
      
      // Step 3: Stop duration timer
      this.stopDurationTimer();
      
      // Step 4: Clear active call
      this.activeCallSubject.next(null);
      
      console.log('[CallState] Call ended');
    } catch (error) {
      console.error('[CallState] Failed to end call:', error);
      throw error;
    }
  }

  /**
   * Start call duration timer
   */
  private startDurationTimer(): void {
    this.callDurationSubject.next(0);
    this.durationInterval = setInterval(() => {
      const current = this.callDurationSubject.value;
      this.callDurationSubject.next(current + 1);
    }, 1000);
  }

  /**
   * Stop call duration timer
   */
  private stopDurationTimer(): void {
    if (this.durationInterval) {
      clearInterval(this.durationInterval);
      this.durationInterval = null;
    }
  }

  /**
   * Get active call
   */
  getActiveCall(): CallSession | null {
    return this.activeCallSubject.value;
  }

  /**
   * Check if there's an active call
   */
  hasActiveCall(): boolean {
    return this.activeCallSubject.value !== null;
  }
}

