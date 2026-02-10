/**
 * HITL Voice Handoff - Alive5 Socket Service
 * Handles Socket.IO connection to Alive5 for incoming call notifications
 */

import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { BehaviorSubject, Observable } from 'rxjs';
import { IncomingCall } from '../models/call.model';

@Injectable({
  providedIn: 'root'
})
export class Alive5SocketService {
  private socket: Socket | null = null;
  private connectedSubject = new BehaviorSubject<boolean>(false);
  public connected$: Observable<boolean> = this.connectedSubject.asObservable();
  
  private incomingCallsSubject = new BehaviorSubject<IncomingCall[]>([]);
  public incomingCalls$: Observable<IncomingCall[]> = this.incomingCallsSubject.asObservable();

  constructor() {}

  /**
   * Upsert an incoming call by room_name (fallback: thread_id).
   * This makes the dashboard idempotent when backend re-emits the same call.
   */
  private upsertIncomingCall(call: IncomingCall): void {
    const currentCalls = this.incomingCallsSubject.value;
    const idx = currentCalls.findIndex(
      (c) => c.room_name === call.room_name || (!!c.thread_id && c.thread_id === call.thread_id),
    );

    if (idx >= 0) {
      const next = [...currentCalls];
      next[idx] = { ...next[idx], ...call };
      this.incomingCallsSubject.next(next);
      return;
    }

    this.incomingCallsSubject.next([...currentCalls, call]);
  }

  /**
   * Get socket instance (for registering custom event listeners)
   */
  getSocket(): Socket | null {
    return this.socket;
  }

  /**
   * Connect to backend Socket.IO server (for HITL dashboard)
   */
  connect(apiKey: string, agentId: string, socketUrl: string = 'http://localhost:8000'): void {
    try {
      console.log('[Alive5Socket] Connecting to backend:', socketUrl);
      
      // Create socket connection to our backend (no auth needed for dashboard)
      this.socket = io(socketUrl, {
        // IMPORTANT:
        // Your current nip.io/nginx setup is not reliably forwarding WebSocket Upgrade headers
        // to /socket.io, which results in backend logs like:
        //   "Invalid websocket upgrade" + 400s
        // Use polling to make the dashboard work everywhere; you can re-enable websocket later
        // once the reverse proxy is configured for Upgrade headers.
        transports: ['polling'],
        upgrade: false,
        path: '/socket.io',
        reconnection: true,
        reconnectionDelay: 2000,  // Wait 2 seconds between attempts
        reconnectionDelayMax: 5000,  // Max 5 seconds
        reconnectionAttempts: 3,  // Only try 3 times
        timeout: 10000  // 10 second connection timeout
      });

      // Set up event listeners
      this.setupEventListeners();
    } catch (error) {
      console.error('[Alive5Socket] Failed to connect:', error);
      throw error;
    }
  }

  /**
   * Set up socket event listeners
   */
  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('[Alive5Socket] Connected');
      this.connectedSubject.next(true);
    });

    this.socket.on('disconnect', (reason: string) => {
      console.log('[Alive5Socket] Disconnected:', reason);
      this.connectedSubject.next(false);
    });

    this.socket.on('connect_error', (error: any) => {
      console.error('[Alive5Socket] Connection error:', error.message || error);
    });

    this.socket.on('reconnect_attempt', (attemptNumber: number) => {
      console.log(`[Alive5Socket] Reconnection attempt ${attemptNumber}/3`);
    });

    this.socket.on('reconnect_failed', () => {
      console.error('[Alive5Socket] Reconnection failed after 3 attempts');
      console.error('[Alive5Socket] Please check that the backend is running and accessible');
    });

    this.socket.on('incoming_human_call', (data: IncomingCall) => {
      console.log('[Alive5Socket] Incoming call:', data);
      // Ensure has_human_agent is set to false for new calls
      if (typeof data.has_human_agent === 'undefined') {
        data.has_human_agent = false;
      }
      this.upsertIncomingCall(data);
    });

    // Listen for human agent joined notifications
    this.socket.on('human_agent_joined', (data: any) => {
      console.log('[Alive5Socket] Human agent joined call:', data);
      
      // Update the call in the incoming list to show it has a human agent
      const currentCalls = this.incomingCallsSubject.value;
      const updatedCalls = currentCalls.map(call => {
        if (call.room_name === data.room_name) {
          return {
            ...call,
            has_human_agent: true,
            human_agent_name: data.agent_name
          };
        }
        return call;
      });
      
      this.incomingCallsSubject.next(updatedCalls);
    });

    // Listen for call ended notifications
    this.socket.on('call_ended', (data: any) => {
      console.log('[Alive5Socket] Call ended:', data);
      
      // Remove the call from the incoming queue
      this.removeIncomingCall(data.room_name);
    });

    this.socket.on('error', (error: any) => {
      console.error('[Alive5Socket] Socket error:', error);
    });
  }

  /**
   * Remove a call from the incoming queue (when accepted or rejected)
   */
  removeIncomingCall(roomName: string): void {
    const currentCalls = this.incomingCallsSubject.value;
    const filteredCalls = currentCalls.filter(call => call.room_name !== roomName);
    this.incomingCallsSubject.next(filteredCalls);
  }

  /**
   * Disconnect from socket
   */
  disconnect(): void {
    if (this.socket) {
      console.log('[Alive5Socket] Disconnecting');
      this.socket.disconnect();
      this.socket = null;
      this.connectedSubject.next(false);
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

